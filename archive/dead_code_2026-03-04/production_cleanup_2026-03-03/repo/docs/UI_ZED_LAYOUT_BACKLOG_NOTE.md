# UI Backlog Note: ZED-Style Trading Workspace

Status: deferred (do not implement now)
Date: 2026-03-02
Priority after paper-trading wiring: high

## Requested layout

- Left sidebar navigation:
  - API
  - Broker
  - Account
  - Instruments
  - Strategies
  - Health/Logs
- Center workspace:
  - Top: tabbed primary views
  - Bottom: tabbed terminal/output panel
- Right sidebar:
  - AI assistant panel
- Styling:
  - Browser-like look and color language
  - Embedded cTrader view as one center tab

## Implementation notes (for later)

- Keep broker/API layer agnostic.
- Reuse central telemetry streams for health, paper/live status, and performance overview.
- Keep terminal tabs split by scope:
  - System health
  - Backtesting
  - Paper trading
  - Live trading
  - Performance summaries
