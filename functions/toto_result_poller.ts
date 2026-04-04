import { broadcast_message } from '@base44/sdk';

/**
 * TOTO Result Poller — Triggered by scheduler every Mon/Thu at 7pm SGT
 * 1. Upgrade EC2 to t3.medium
 * 2. SSH in and run post_draw_pipeline.sh (scrape + check wins + retrain + predict)
 * 3. Spin EC2 back down to t3.nano
 * 4. Message Bharath with results + new predictions
 *
 * SSH KEY FIX (permanent): Key is written from EC2_SSH_KEY env var at runtime.
 * Never relies on /tmp/statlot.pem persisting between runs.
 */

const INSTANCE_ID = 'i-03b425c081ae812b8';
const REGION = 'ap-southeast-1';
const EC2_HOST = '3.1.133.166';
const IDLE_TYPE = 't3.nano';
const ACTIVE_TYPE = 't3.medium';
const PEM_PATH = '/tmp/statlot.pem';

async function restore_ssh_key(): Promise<string> {
  const key = process.env.EC2_SSH_KEY;
  if (!key) throw new Error('EC2_SSH_KEY env var not set');

  const fs = await import('fs');
  const os = await import('os');
  const { execSync } = await import('child_process');

  const header = '-----BEGIN RSA PRIVATE KEY-----';
  const footer = '-----END RSA PRIVATE KEY-----';

  // Strip headers/whitespace, rewrap at 64 chars
  let body = key
    .replace(header, '')
    .replace(footer, '')
    .replace(/\s+/g, '');

  const lines: string[] = [];
  for (let i = 0; i < body.length; i += 64) {
    lines.push(body.slice(i, i + 64));
  }
  const pem = `${header}\n${lines.join('\n')}\n${footer}\n`;

  fs.writeFileSync(PEM_PATH, pem, { mode: 0o400 });
  console.log(`[SSH KEY] Restored to ${PEM_PATH}`);
  return PEM_PATH;
}

async function ec2_scale(newType: string): Promise<void> {
  const boto3 = await import('boto3' as any).catch(() => null);
  // Use AWS SDK v3 (available in Deno/Node)
  const { EC2Client, StopInstancesCommand, StartInstancesCommand,
          ModifyInstanceAttributeCommand, DescribeInstancesCommand } = await import('@aws-sdk/client-ec2');

  const client = new EC2Client({ region: REGION });

  const currentState = async () => {
    const resp = await client.send(new DescribeInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
    return resp.Reservations![0].Instances![0].State!.Name;
  };

  const waitForState = async (target: string, maxWait = 300) => {
    const start = Date.now();
    while (Date.now() - start < maxWait * 1000) {
      const state = await currentState();
      console.log(`[EC2] State: ${state} (waiting for ${target})`);
      if (state === target) return;
      await new Promise(r => setTimeout(r, 10000));
    }
    throw new Error(`Timed out waiting for EC2 state: ${target}`);
  };

  console.log(`[EC2] Stopping instance for resize to ${newType}...`);
  await client.send(new StopInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  await waitForState('stopped');

  await client.send(new ModifyInstanceAttributeCommand({
    InstanceId: INSTANCE_ID,
    InstanceType: { Value: newType }
  }));
  console.log(`[EC2] Instance type set to ${newType}`);

  await client.send(new StartInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  await waitForState('running');
  console.log(`[EC2] Instance running as ${newType} ✅`);

  // Wait for SSH
  await new Promise(r => setTimeout(r, 30000));
}

async function run_pipeline_ssh(pemPath: string): Promise<{ success: boolean; log: string }> {
  try {
    const { Client } = await import('ssh2');
    const fs = await import('fs');
    const privateKey = fs.readFileSync(pemPath);

    return new Promise((resolve) => {
      const conn = new Client();
      let log = '';

      conn.on('ready', () => {
        console.log('[SSH] Connected to EC2');
        conn.exec(
          'cd ~/statlot-649/statlot && source venv/bin/activate && python3 -m toto.retrain_and_predict 2>&1',
          (err, stream) => {
            if (err) {
              conn.end();
              resolve({ success: false, log: `exec error: ${err}` });
              return;
            }
            stream.on('data', (data: Buffer) => { log += data.toString(); });
            stream.stderr.on('data', (data: Buffer) => { log += data.toString(); });
            stream.on('close', (code: number) => {
              conn.end();
              resolve({ success: code === 0, log });
            });
          }
        );
      });

      conn.on('error', (err) => {
        resolve({ success: false, log: `SSH connection error: ${err}` });
      });

      conn.connect({
        host: EC2_HOST,
        username: 'ubuntu',
        privateKey,
        readyTimeout: 30000,
      });
    });
  } catch (error) {
    return { success: false, log: `SSH error: ${error}` };
  }
}

export async function toto_result_poller(context: any) {
  console.log(`[TOTO PIPELINE] Starting at ${new Date().toISOString()}`);

  // Step 1: Restore SSH key from env var — NEVER relies on /tmp/statlot.pem persisting
  let pemPath: string;
  try {
    pemPath = await restore_ssh_key();
    console.log('[SSH KEY] ✅ Key restored from EC2_SSH_KEY env var');
  } catch (err) {
    const msg = `❌ SSH key restore failed: ${err}. Pipeline aborted.`;
    console.error(msg);
    await broadcast_message({ message: msg, channels: ['telegram', 'web'] });
    return { success: false, error: String(err) };
  }

  // Step 2: Scale up to t3.medium
  try {
    await ec2_scale(ACTIVE_TYPE);
  } catch (err) {
    const msg = `❌ EC2 scale-up to ${ACTIVE_TYPE} failed: ${err}. Pipeline aborted.`;
    await broadcast_message({ message: msg, channels: ['telegram', 'web'] });
    return { success: false, error: String(err) };
  }

  // Step 3: Run pipeline
  console.log('[PIPELINE] Running retrain + predict on EC2...');
  const result = await run_pipeline_ssh(pemPath);

  // Step 4: Scale back to t3.nano — always, even if pipeline failed
  try {
    await ec2_scale(IDLE_TYPE);
    console.log(`[EC2] ✅ Scaled back to ${IDLE_TYPE}`);
  } catch (err) {
    console.error(`[EC2] ⚠️ Scale-down failed: ${err} — manual intervention needed`);
  }

  // Step 5: Report to Bharath
  const status = result.success ? '✅ Pipeline completed' : '⚠️ Pipeline had errors';
  const lastLines = result.log.split('\n').slice(-20).join('\n');

  const message = `🎯 **TOTO Pipeline** — ${new Date().toLocaleString('en-SG', { timeZone: 'Asia/Singapore' })}

**Status:** ${status}
**EC2:** Scaled back to ${IDLE_TYPE}

**Last output:**
\`\`\`
${lastLines}
\`\`\`

Check DuckDB toto_predictions_log for new predictions.`;

  await broadcast_message({ message, channels: ['telegram', 'web'] });
  return { success: result.success, log: result.log };
}
