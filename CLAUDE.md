# Kinetra — Agent Rules

## cTrader Open API

**Official docs**: https://help.ctrader.com/open-api/
**Proto messages**: https://github.com/spotware/openapi-proto-messages/blob/main/OpenApiModelMessages.proto

### ProtoOAExecutionType enum values

Always verify integer constants against the proto source above. The correct values are:

| Name                  | Value |
|-----------------------|-------|
| ORDER_ACCEPTED        | 2     |
| ORDER_FILLED          | 3     |
| ORDER_REPLACED        | 4     |
| ORDER_CANCELLED       | 5     |
| ORDER_EXPIRED         | 6     |
| ORDER_REJECTED        | 7     |
| ORDER_CANCEL_REJECTED | 8     |
| SWAP                  | 9     |
| DEPOSIT_WITHDRAW      | 10    |
| ORDER_PARTIAL_FILL    | 11    |

**Never guess or derive these from context.** Off-by-one errors here cause silent fill event drops and live order timeouts.

### ProtoOANewOrderReq has no synchronous response

`ProtoOANewOrderReq` does **not** return a matching RPC response over the request/response channel. The fill is delivered asynchronously as a `ProtoOAExecutionEvent` push. Consequences:

- `send_and_wait(req)` with the default 10 s timeout fires on every order and increments `request_timeout_count`, eventually triggering reconnect cascades.
- Use `send_and_wait(req, timeout_s=3.0)` only to catch an immediate `ProtoOAErrorRes` rejection. A `None` return is normal and means no immediate error.

### Reactor thread must never block

cTrader callbacks (`_on_spot_event`, `_on_execution_event`, etc.) fire on the Twisted reactor thread. Any blocking call on that thread stalls all I/O — fills never arrive, heartbeats are missed, and subscriptions become silent.

**Rule**: bar-processing callbacks must enqueue work onto a `queue.Queue` and return immediately. A dedicated worker thread dequeues and calls blocking engine/order logic (`process_bar`, `send_and_wait`, etc.).

```python
# reactor thread — must not block
def _on_bar(symbol, close, timestamp, **_):
    try:
        _bar_queue.put_nowait((close, timestamp))
    except queue.Full:
        LOG.warning("Bar queue full — dropping bar")

# worker thread — safe to block
def _worker():
    while True:
        item = _bar_queue.get()
        if item is _STOP:
            break
        close, ts = item
        engine.process_bar(close, ts)  # may call send_and_wait
```

### Rich Live display

Do not call `click.echo` / `click.secho` after `live_ctx.start()`. Those writes bypass the Rich live context and corrupt the in-place frame refresh. Move all informational prints to **before** `live_ctx.start()`.
