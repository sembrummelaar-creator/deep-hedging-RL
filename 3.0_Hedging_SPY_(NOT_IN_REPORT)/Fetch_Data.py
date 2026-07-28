import os
import yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))
TICKER = "SPY"
N_POINTS = 64          # N+1 = 63 steps + 1

# Pull ~5 months of daily closes, keep the last N_POINTS
df = yf.download(TICKER, period="5mo", interval="1d", auto_adjust=True)
closes = df["Close"].dropna().values.flatten()[-N_POINTS:]

assert len(closes) == N_POINTS, f"got {len(closes)} points, need {N_POINTS}"

import numpy as np
np.savetxt(os.path.join(OUT, "real_path.csv"), closes, delimiter=",")
print(f"Saved {N_POINTS} closes for {TICKER}: "
      f"start={closes[0]:.2f}, end={closes[-1]:.2f}")