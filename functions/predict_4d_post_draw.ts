import { broadcast_message } from '@base44/sdk';

/**
 * 4D Post-Draw Pipeline — Triggered by scheduler every Wed/Sat/Sun at 7pm SGT
 *
 * Full pipeline on EC2 (in order):
 *   1. git pull (pick up latest script fixes)
 *   2. python3 -m 4d.scrape_latest     → updates latest_draw.json
 *   3. python3 -m 4d.check_wins        → reads predictions_log, writes results_log
 *   4. python3 -m 4d.retrain_and_predict → trains on all draws, writes predictions_log
 *
 * EC2 lifecycle: t3.nano → t3.medium → (pipeline) → t3.nano
 * SSH key: always restored from EC2_SSH_KEY env var at runtime
 */

const INSTANCE_ID = 'i-03b425c081ae812b8';
const REGION      = 'ap-southeast-1';
const EC2_HOST    = '3.1.133.166';
const IDLE_TYPE   = 't3.nano';
const ACTIVE_TYPE = 't3.medium';
const PEM_PATH    = '/tmp/statlot_4d.pem';
const VENV_PY     = '/home/ubuntu/statlot-649/statlot/venv/bin/python3';
const WORK_DIR    = '/home/ubuntu/statlot-649/statlot';

async function restore_ssh_key(): Promise<string> {
  const key = process.env.EC2_SSH_KEY;
  if (!key) throw new Error('EC2_SSH_KEY env var not set');
  const fs = await import('fs');
  const header = '-----BEGIN RSA PRIVATE KEY-----';
  const footer  = '-----END RSA PRIVATE KEY-----';
  let body = key.replace(header, '').replace(footer, '').replace(/\s+/g, '');
  const lines: string[] = [];
  for (let i = 0; i < body.length; i += 64) lines.push(body.slice(i, i + 64));
  const pem = `${header}\n${lines.join('\n')}\n${footer}\n`;
  fs.writeFileSync(PEM_PATH, pem, { mode: 0o400 });
  console.log(`[SSH KEY] Restored to ${PEM_PATH}`);
  return PEM_PATH;
}

async function ec2_scale(newType: string): Promise<void> {
  const { EC2Client, StopInstancesCommand, StartInstancesCommand,
          ModifyInstanceAttributeCommand, DescribeInstancesCommand } =
    await import('@aws-sdk/client-ec2');
  const client = new EC2Client({ region: REGION });
  const getState = async () => {
    const resp = await client.send(new DescribeInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
    return resp.Reservations![0].Instances![0].State!.Name;
  };
  const waitFor = async (target: string, maxSec = 360) => {
    const deadline = Date.now() + maxSec * 1000;
    while (Date.now() < deadline) {
      const s = await getState();
      console.log(`[EC2] state=${s} (want ${target})`);
      if (s === target) return;
      await new Promise(r => setTimeout(r, 10_000));
    }
    throw new Error(`Timeout waiting for EC2 state: ${target}`);
  };
  const resp0 = await client.send(new DescribeInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  const inst0  = resp0.Reservations![0].Instances![0];
  if (inst0.InstanceType === newType && inst0.State!.Name === 'running') {
    console.log(`[EC2] Already ${newType} and running — skip resize`);
    return;
  }
  console.log(`[EC2] Stopping for resize to ${newType}...`);
  await client.send(new StopInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  await waitFor('stopped');
  await client.send(new ModifyInstanceAttributeCommand({
    InstanceId: INSTANCE_ID, InstanceType: { Value: newType },
  }));
  await client.send(new StartInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  await waitFor('running');
  await new Promise(r => setTimeout(r, 20_000));
  console.log(`[EC2] Now running as ${newType}`);
}

async function ssh_run(pem: string, cmd: string, timeoutMs: number): Promise<string> {
  const { execSync } = await import('child_process');
  const full = `ssh -i ${pem} -o StrictHostKeyChecking=no -o ConnectTimeout=30 ubuntu@${EC2_HOST} "${cmd}"`;
  const out = execSync(full, { timeout: timeoutMs, encoding: 'utf8' });
  return out;
}

export async function predict_4d_post_draw(input: any) {
  console.log('[4D PIPELINE] Starting post-draw pipeline');
  const pem = await restore_ssh_key();
  let summary = '';

  try {
    // Scale up to t3.medium
    await ec2_scale(ACTIVE_TYPE);

    // Step 1 — git pull
    console.log('[4D] Step 1: git pull');
    await ssh_run(pem, `cd ~/statlot-649 && git pull`, 60_000);

    // Step 2 — scrape latest 4D result
    console.log('[4D] Step 2: scrape_latest');
    const scrape_out = await ssh_run(pem,
      `cd ${WORK_DIR} && ${VENV_PY} -m 4d.scrape_latest 2>&1`,
      120_000);
    console.log('[4D scrape]', scrape_out);

    // Step 3 — check wins
    console.log('[4D] Step 3: check_wins');
    const wins_out = await ssh_run(pem,
      `cd ${WORK_DIR} && ${VENV_PY} -m 4d.check_wins 2>&1`,
      120_000);
    console.log('[4D wins]', wins_out);
    summary += `Win check:\n${wins_out}\n\n`;

    // Step 4 — retrain and predict
    console.log('[4D] Step 4: retrain_and_predict');
    const retrain_out = await ssh_run(pem,
      `cd ${WORK_DIR} && ${VENV_PY} -m 4d.retrain_and_predict 2>&1`,
      900_000);
    console.log('[4D retrain]', retrain_out);
    summary += `New predictions:\n${retrain_out}\n`;

    await broadcast_message({ message: `[4D] Pipeline complete ✅\n\n${summary}` });

  } catch (err: any) {
    console.error('[4D PIPELINE ERROR]', err.message);
    await broadcast_message({ message: `[4D] Pipeline ERROR ❌\n${err.message}` });
  } finally {
    // Always scale back to nano
    try {
      await ec2_scale(IDLE_TYPE);
      console.log('[4D] EC2 back to t3.nano ✅');
    } catch (e: any) {
      console.error('[4D] Scale-down failed:', e.message);
    }
  }
}
