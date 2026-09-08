# September 8 readiness and budget safety

Authenticated account reads at 02:09 and 02:15 Europe/London agree:
credits $245, cumulative usage $220.376693994, balance $24.623306006.
Usage is unchanged from the last saved boundary. No additional top-up has posted.
The September 8 ledger freezes this boundary with a $5 account-wide cap and
zero recorded spend. No paid inference calls were made or authorized.

The authenticated catalog at 02:09:14 lists both exact requested routes:

| Model | Input / million tokens | Output / million tokens |
|---|---:|---:|
| deepseek/deepseek-v4-flash-0731 | $0.14 | $0.28 |
| openai/gpt-5.6-luna | $0.20 | $1.20 |

These supersede the older promotional prices for future budgeting. Luna lists
a 272,000-token input threshold with $0.40/$1.80 rates; cache and search pricing
are separate. Catalog presence and structured-output support do not establish
serving validity or semantic calibration. Re-read exact prices before any paid
block; reserve its full worst-case exposure, not an assumed average/cache discount.

## Safety correction

`scripts/openrouter_daily_budget.py` now rejects nonfinite, negative, boolean,
and string monetary inputs; non-London ledgers; naive timestamps; caps above $5
or at zero; and cumulative usage below the frozen opening. Decimal comparisons
remove the previous tolerance that could expand the cap. Reconciliation still
uses the larger of posted usage and locally recorded accepted-request cost.
All 36 focused budget tests pass.

This helper is only a numeric daily-allowance check. It is not an atomic
in-flight reservation ledger, sufficient-balance check, scientific authorization,
or concurrent dispatcher. Those checks remain mandatory before actual requests.
No frozen historical experiment is rerun or reauthorized by this change.

## Research consequence

The complete fresh proposer-plus-predictive-weight protocol remains the next
dependency. The retrospective 0.78% transfer gain motivates that test but cannot
serve as its confirmation or as a non-myopic efficacy claim. Do not launch a depth
sweep based merely on account balance or catalog availability. Automation stays
paused and the overall research objective remains unfinished.
