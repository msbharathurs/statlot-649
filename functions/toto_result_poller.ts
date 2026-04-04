import { broadcast_message } from '@base44/sdk';

/**
 * TOTO Result Poller — Triggered by scheduler every Mon/Thu at 7pm SGT
 *
 * Full pipeline on EC2 (in order):
 *   1. git pull (pick up latest script fixes)
 *   2. python3 -m toto.scrape_latest   → updates latest_draw.json
 *   3. python3 -m toto.check_wins      → reads toto_predictions_log, writes toto_results
 *   4. python3 -m toto.retrain_and_predict → trains on all draws, writes toto_predictions_log
 *
 * EC2 lifecycle: t3.nano → t3.medium → (pipeline) → t3.nano
 * SSH key: always restored from EC2_SSH_KEY env var at runtime (never from /tmp)
 */

const INSTANCE_ID = 'i-03b425c081ae812b8';
const REGION      = 'ap-southeast-1';
const EC2_HOST    = '3.1.133.166';
const IDLE_TYPE   = 't3.nano';
const ACTIVE_TYPE = 't3.medium';
const PEM_PATH    = '/tmp/statlot_toto.pem';

// ── SSH key restore ───────────────────────────────────────────────────────────
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

// ── EC2 scale ────────────────────────────────────────────────────────────────
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

  // Skip stop/resize if already the right type AND running
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
  console.log(`[EC2] Set type → ${newType}`);

  await client.send(new StartInstancesCommand({ InstanceIds: [INSTANCE_ID] }));
  await waitFor('running');

  // Give SSH daemon ~30s to come up
  await new Promise(r => setTimeout(r, 30_000));
  console.log(`[EC2] ✅ Running as ${newType}`);
}

// ── SSH exec helper ──────────────────────────────────────────────────────────
async function ssh_exec(
  pemPath: string,
  command: string,
  timeoutMs = 900_000,   // 15 min default
): Promise<{ success: boolean; stdout: string }> {
  const { Client } = await import('ssh2');
  const fs = await import('fs');
  const privateKey = fs.readFileSync(pemPath);

  return new Promise((resolve) => {
    const conn = new Client();
    let out = '';
    const timer = setTimeout(() => {
      conn.end();
      resolve({ success: false, stdout: out + '\n[TIMEOUT after 15min]' });
    }, timeoutMs);

    conn.on('ready', () => {
      conn.exec(command, (err, stream) => {
        if (err) {
          clearTimeout(timer);
          conn.end();
          resolve({ success: false, stdout: `exec error: ${err}` });
          return;
        }
        stream.on('data', (d: Buffer) => { out += d.toString(); });
        stream.stderr.on('data', (d: Buffer) => { out += d.toString(); });
        stream.on('close', (code: number) => {
          clearTimeout(timer);
          conn.end();
          resolve({ success: code === 0, stdout: out });
        });
      });
    });

    conn.on('error', (err) => {
      clearTimeout(timer);
      resolve({ success: false, stdout: `SSH error: ${err}` });
    });

    conn.connect({
      host: EC2_HOST, username: 'ubuntu',
      privateKey, readyTimeout: 30_000,
    });
  });
}

// ── Main entry point ─────────────────────────────────────────────────────────
export async function toto_result_poller(context: any) {
  console.log(`[TOTO PIPELINE] Starting — ${new Date().toISOString()}`);

  // ── Step 0: Restore SSH key ──────────────────────────────────────────────
  let pemPath: string;
  try {
    pemPath = await restore_ssh_key();
  } catch (err) {
    const msg = `❌ TOTO Pipeline: SSH key restore failed — ${err}. Aborted.`;
    await broadcast_message({ message: msg, channels: ['telegram', 'web'] });
    return { success: false, error: String(err) };
  }

  // ── Step 1: Scale up to t3.medium ───────────────────────────────────────
  try {
    await ec2_scale(ACTIVE_TYPE);
  } catch (err) {
    const msg = `❌ TOTO Pipeline: EC2 scale-up failed — ${err}. Aborted.`;
    await broadcast_message({ message: msg, channels: ['telegram', 'web'] });
    return { success: false, error: String(err) };
  }

  // ── Step 2: git pull (pick up latest script fixes from GitHub) ──────────
  console.log('[STEP 2] git pull...');
  const pull = await ssh_exec(pemPath,
    'cd ~/statlot-649 && git pull origin main 2>&1', 60_000);
  console.log('[git pull]', pull.stdout.slice(-200));

  // ── Step 3: scrape_latest.py ─────────────────────────────────────────────
  console.log('[STEP 3] scrape_latest.py...');
  const scrape = await ssh_exec(pemPath,
    'cd ~/statlot-649/statlot && /home/ubuntu/statlot-649/venv/bin/python3 -m toto.scrape_latest 2>&1',
    120_000);
  console.log('[scrape]', scrape.stdout);

  if (!scrape.success) {
    // Don't abort — result may still be available from a previous scrape run
    console.warn('[scrape] Non-zero exit. Continuing with existing latest_draw.json.');
  }

  // ── Step 4: check_wins.py (reads toto_predictions_log) ───────────────────
  console.log('[STEP 4] check_wins.py...');
  const wins = await ssh_exec(pemPath,
    'cd ~/statlot-649/statlot && /home/ubuntu/statlot-649/venv/bin/python3 -m toto.check_wins 2>&1',
    120_000);
  console.log('[check_wins]', wins.stdout);

  // ── Step 5: retrain_and_predict.py ───────────────────────────────────────
  console.log('[STEP 5] retrain_and_predict.py...');
  const retrain = await ssh_exec(pemPath,
    'cd ~/statlot-649/statlot && /home/ubuntu/statlot-649/venv/bin/python3 -m toto.retrain_and_predict 2>&1',
    900_000);
  console.log('[retrain]', retrain.stdout);

  // ── Step 6: Scale back to t3.nano (always — even if pipeline failed) ─────
  try {
    await ec2_scale(IDLE_TYPE);
    console.log(`[EC2] ✅ Scaled back to ${IDLE_TYPE}`);
  } catch (err) {
    console.error(`[EC2] ⚠️ Scale-down failed: ${err} — MANUAL INTERVENTION NEEDED`);
  }

  // ── Step 7: Parse prediction output for summary ───────────────────────────
  const retrain_log  = retrain.stdout;
  const wins_log     = wins.stdout;

  // Extract key lines from retrain output
  const pred_lines = retrain_log.split('\n').filter(l =>
    l.includes('Sys6') || l.includes('Sys7') || l.includes('Bonus') ||
    l.includes('PREDICTION SAVED') || l.includes('Training draws') ||
    l.includes('Additional picks') || l.includes('Cost mandatory')
  ).join('\n');

  // Extract win summary from check_wins output
  const win_lines = wins_log.split('\n').filter(l =>
    l.includes('Winning:') || l.includes('Best group') ||
    l.includes('Total prize') || l.includes('Any win') ||
    l.includes('Checking draw')
  ).join('\n');

  const overall_success = retrain.success;
  const status_icon = overall_success ? '✅' : '⚠️';

  const sgTime = new Date().toLocaleString('en-SG', { timeZone: 'Asia/Singapore' });

  const message =
`🎯 **TOTO Post-Draw Pipeline** — ${sgTime}

**Status:** ${status_icon} ${overall_success ? 'All steps completed' : 'Pipeline had errors — check logs'}
**EC2:** Scaled back to ${IDLE_TYPE}

**Draw result + win check:**
\`\`\`
${win_lines || '(no output)'}
\`\`\`

**New predictions:**
\`\`\`
${pred_lines || '(no output — check EC2 logs)'}
\`\`\``;

  await broadcast_message({ message, channels: ['telegram', 'web'] });

  return {
    success: overall_success,
    scrape_ok: scrape.success,
    wins_ok:   wins.success,
    retrain_ok: retrain.success,
  };
}
