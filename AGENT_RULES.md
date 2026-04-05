# AGENT_RULES.md
# Statlot Operating Rules — Base44
# Last updated: 2026-04-05
# READ THIS BEFORE TOUCHING ANYTHING. EVERY SESSION. NO EXCEPTIONS.

---

## GIT PUSH RULE — NON-NEGOTIABLE

Every commit must be on GitHub. EC2 local commits do not exist.

The done definition already requires git push. This rule makes it explicit:

- `git commit` alone is NOT done
- `git commit + git push` is done
- If push fails, the task is NOT complete — fix the push before moving on
- Never end a session with EC2 ahead of origin/main by any commits
- Never say "commit is on EC2 local git" as if that is acceptable
- EC2 can be terminated, snapshotted, or lost at any time
- GitHub is the only permanent record

If the GitHub token is expired or push credentials fail:
1. Stop immediately and report it
2. Fix the token (generate new PAT at github.com/settings/tokens)
3. Push before doing anything else
4. Never defer a push to "next session"

Verify at session end:
```bash
cd ~/statlot-649 && git status && git log origin/main..HEAD
```
If that shows any commits — PUSH THEM before ending the session.

---

## SESSION START RITUAL (mandatory — in this exact order)

Before any work, before any code, before answering any question:

### Step 1 — Load state
```bash
cat ~/statlot-649/AGENT_STATE.md
```
Read it fully. This is the ground truth of what exists and what is broken.

### Step 2 — Verify ground truth
```bash
# Confirm the canonical DuckDB is intact
python3 -c "
import duckdb
con = duckdb.connect('/home/ubuntu/statlot-649/statlot_toto.duckdb')
for row in con.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema='main'\").fetchall():
    t = row[0]
    n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {n} rows')
con.close()
"
# Confirm git is clean and up to date
cd ~/statlot-649 && git status && git log --oneline -5
```

### Step 3 — Report before proceeding
State out loud (in your response to Bharath):
- What AGENT_STATE.md says is in progress
- What the DB row counts show right now
- Any mismatch between state and reality
- Then ask: "What do you want to work on?"

**Do NOT skip this even if Bharath has already told you what to do.**
**The ritual happens first.**

---

## CANONICAL FILE PATHS — DO NOT DEVIATE

These are the only correct paths. Never create new DB files. Never write scripts
to a sandbox folder and assume they'll be there next session.

| Resource              | Canonical Path |
|-----------------------|----------------|
| TOTO DuckDB           | /home/ubuntu/statlot-649/statlot_toto.duckdb |
| 4D DuckDB             | /home/ubuntu/statlot-649/draws_4d.duckdb |
| TOTO scripts          | ~/statlot-649/statlot/toto/ |
| Engine models         | ~/statlot-649/statlot/engine/models/ |
| Historical draws JSON | ~/statlot-649/statlot/sp_historical_draws.json |
| Backtest scripts      | ~/statlot-649/statlot/ |
| Agent state           | ~/statlot-649/AGENT_STATE.md |
| Agent rules           | ~/statlot-649/AGENT_RULES.md |

**If you find yourself writing to a path not in this table — STOP.**
Either the path is wrong or the table needs updating.
Do not silently create new files in new locations.

---

## THE DONE DEFINITION

Nothing is "done" until ALL three of these are true:

1. **Code is committed and pushed to GitHub**
```bash
cd ~/statlot-649 && git add -A && git commit -m "describe what changed" && git push
```

2. **Actual EC2 output is pasted as evidence**
   - Not "I ran the command and it worked"
   - Not "tests passed"
   - The actual terminal output, pasted verbatim

3. **AGENT_STATE.md is updated** to reflect what changed

If any of these three are missing, the task is not done.
"I'll update AGENT_STATE.md at the end" is not acceptable — update it now, before moving to the next task.

---

## FORBIDDEN ACTIONS

Never do these without explicit instruction from Bharath:
- Change EC2 instance type (nano → medium or any other size)
- Drop or truncate any DuckDB table
- Delete files from git history
- Write pipeline scripts to `.agents/scripts/` — they will be lost next session
- Overwrite `sp_historical_draws.json` without first backing it up
- Run a retrain without verifying the draw data completeness first
- Mark an automation as "working" without running a dry-run on EC2 and seeing actual output
- Create a new DuckDB file anywhere — use the canonical paths above

---

## AUTOMATION HONESTY RULE

An automation status of "success" means the automation did not crash.
It does NOT mean the actual work (scraping, retraining, predicting) happened.

Before claiming an automation works:
1. SSH to EC2 and run the pipeline script manually
2. Paste the actual output
3. Verify the DB has new rows after it ran
4. Only then say "the automation works"

---

## DATA VERIFICATION RULE

Before writing any code that reads from a DB, JSON file, or external source:
1. Query or read the actual data first
2. Check for NULL values, missing rows, corrupt dates
3. State what you found — do not silently handle NULLs with "safe defaults"

A model trained on NULL data or missing 1,828 draws is not a model.
A "safe default of 50.0 when NULL" is hiding a broken data pipeline.

---

## SESSION END RITUAL (mandatory)

Before ending any session:

1. **Update AGENT_STATE.md** with:
   - What was completed (with evidence — row counts, git commit hash)
   - What is still in progress and exactly where it was left
   - Any new broken things discovered
   - Next steps (specific, not vague)

2. **Commit and push everything to GitHub**
```bash
cd ~/statlot-649 && git add -A && git commit -m "session end: [describe state]" && git push
```

3. **Verify no local-only commits remain:**
```bash
git log origin/main..HEAD
```
If this shows anything — stop and push before ending.

4. **Confirm to Bharath:** "Session end — AGENT_STATE.md updated, everything pushed to git. Commit hash: [hash]. Here is what is in progress and what is next."

If the session ends without this, the next session starts blind.
Every session that starts blind wastes Bharath's time and money.

---

## SELF-HONESTY RULE

If something is broken, say so immediately.
Do not say "let me fix it" and then silently build around the broken thing.
Do not say "automation ran successfully" when you mean "the automation did not crash."
Do not cite backtest numbers from a model trained on incomplete data.

Bharath's exact words: "your honesty doesn't fix any shit."
The goal is not honest reporting of failure. The goal is not failing.
The honest reporting is only useful when paired with: "here is what I am doing right now to fix it."
