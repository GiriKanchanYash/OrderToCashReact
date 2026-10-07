# Data sources: Snowflake and Microsoft Fabric

Users pick the data source from the **Data source** selector in the top bar. The choice is stored
in the `o2c_data_source` cookie and the page reloads; every `/api` call is then routed to that source.

## Run

    uvicorn server:app --port 8001        # local
    startup.sh                             # Azure App Service (now starts server:app)

`server.py` wraps the two apps without modifying them:

| Path | Handled by |
|---|---|
| `/api/*` | the selected source (`X-Data-Source` header > cookie > `DEFAULT_DATA_SOURCE`) |
| `/api/data-source` | the gateway (current selection + which sources loaded) |
| `/health/fabric` | Fabric app (round-trips `SELECT 1`) |
| everything else (UI, `/assets`, `/health`) | Snowflake app, exactly as before |

If the Fabric stack can't load (for example, the ODBC driver is missing), Snowflake keeps working
and Fabric requests return a 503 that explains why.

## How Fabric mirrors Snowflake

`app/` (Snowflake) is unchanged. `FabicApp/services/*` are copies of `app/services/*` with the
same functions, queries and response shapes, so behaviour stays in step:

* **SQL**: `FabicApp/sql_dialect.py` translates each Snowflake-dialect query to T-SQL at run time
  (sqlglot, plus fixes for integer AVG/division, CURRENT_DATE, Monday-start weeks, GROUP BY
  ordinals/aliases, ORDER BY in CTEs, ISO date strings, and MERGE terminators).
  `weekly_forecast.py` is hand-written T-SQL.
* **AI**: Azure OpenAI replaces Cortex COMPLETE and Cortex Agents (`agent_runtime.py`).
  `fabric_analyst.py` replaces Cortex Analyst; its generated SQL is checked to be read-only and
  limited to BUSINESS_MART views before it runs. Policy search uses `O2C_AGENT.POLICY_KB`.
* **Writes** (PTP, contacts, cash apply, disputes) go to `SAP_STG`, then
  `EXEC RAW_VAULT.SP_LOAD_*_DOMAIN`.

When you change a Snowflake service, copy the change to the matching `FabicApp/services` file.
Snowflake-dialect SQL is translated automatically.

## Fabric prerequisites

* The `OrderToCash_DW` warehouse contains `BUSINESS_MART`, `INFORMATION_MART`, `SAP_STG`,
  `RAW_VAULT` and `O2C_AGENT`, with the same object names as Snowflake. If they differ,
  set `FABRIC_SCHEMA_MAP` / `FABRIC_IDENTIFIER_CASE`.
* The service principal can read the warehouse and write to `SAP_STG`, `INFORMATION_MART.SAVED_INSIGHTS`
  and `GENIE_QUESTION_HISTORY`.
* ODBC Driver 18 for SQL Server is installed on the host.
* Optional: put your semantic model in `FabicApp/Schema_model.yml` to improve Copilot SQL.

## Frontend source

The selector was patched into the compiled bundle (`static/assets/index-Dh8OojcJ.js`) and
`static/index.html`. **Re-apply it in your frontend source before the next build**, or the
build will drop it. Add the selection script and styles from `static/index.html`, and add
this component to the app shell's `<header className="app-topbar">`, just before the logo `<img>`:

```tsx
function DataSourceSelect() {
  const [value, setValue] = useState<string>((window as any).__O2C_DS__ || "snowflake");
  const [info, setInfo] = useState<any>(null);
  useEffect(() => {
    fetch("/api/data-source").then(r => (r.ok ? r.json() : null)).then(d => {
      if (!d) return;
      setInfo(d);
      if (!(window as any).__O2C_DS__) { (window as any).__O2C_DS__ = d.selected; setValue(d.selected); }
    }).catch(() => {});
  }, []);
  const sources = info?.sources ?? [
    { id: "snowflake", label: "Snowflake", available: true },
    { id: "fabric", label: "Microsoft Fabric", available: true },
  ];
  return (
    <label className="datasource-picker" title="Choose the data source for all pages">
      <span className="datasource-picker-label">Data source</span>
      <select className="datasource-picker-select" aria-label="Data source" value={value}
        onChange={e => { const v = e.target.value; if (v === value) return;
          (window as any).__O2C_SET_DS__(v); setValue(v); window.location.reload(); }}>
        {sources.map((s: any) => (
          <option key={s.id} value={s.id} disabled={!s.available && s.id !== value}>
            {s.label}{s.available ? "" : " (unavailable)"}
          </option>
        ))}
      </select>
    </label>
  );
}
```

Also make these labels source-aware, as in the bundle: "Loading KPIs from Snowflake…",
"Running Snowflake Cortex agent…" (x4), "Snowflake Agent · unavailable", and the
`"Snowflake Agent"` fallback in the agent badge.
