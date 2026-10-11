# Using the MCP proxy with other clients

`noulgate mcp-proxy` wraps any **stdio** MCP server. Point your client at the proxy instead of the
server and put the original command after `--`:

```bash
noulgate mcp-proxy --backend rules --policy policy.json -- npx -y @modelcontextprotocol/server-filesystem ~/work
```

To share one resident model across proxies, run `noulgate serve --backend d1-3b --policy policy.json`
and pass `--server http://127.0.0.1:7701` to each proxy.

## What was tested

`tests/test_mcp_proxy.py` drives the proxy with the official MCP Python SDK `ClientSession` (v2.2).
That is a non-Claude-Code MCP client, which is what M4 requires. It checks:

- `tools/list` passthrough;
- an allowed call is forwarded;
- an exfiltration call whose target only appeared in an earlier tool result is blocked;
- `ask` becomes an MCP form elicitation when the client declares the capability, and fails closed
  when it does not.

GUI clients were **not** tested in this environment. The configurations below follow each client's
documented `mcpServers` format, but verify them yourself.

### Cursor (`~/.cursor/mcp.json`) / Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "filesystem-gated": {
      "command": "noulgate",
      "args": ["mcp-proxy", "--policy", "/path/policy.json", "--",
               "npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/me/work"]
    }
  }
}
```

### VS Code (`.vscode/mcp.json`)

```json
{
  "servers": {
    "filesystem-gated": {
      "type": "stdio",
      "command": "noulgate",
      "args": ["mcp-proxy", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    }
  }
}
```

## Limits

- MCP does not tell the server what the user asked for, so the proxy sees no `user` spans. Every
  target that came from an earlier tool result counts as "only in untrusted content", and
  `user_requested` carries less signal than in the Claude Code hook. Calibrate the proxy on proxy
  traces, not on hook traces: the score distributions differ.
- Only tools are gated. Resources and prompts are not proxied.
- The proxy does not support streamable HTTP upstreams.
