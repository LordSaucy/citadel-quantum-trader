import EmbeddedGrafana from "./EmbeddedGrafana";
import ArbExecutor from "./ArbExecutor";
import EmbeddedGrafana from "./EmbeddedGrafana";
import React, { useEffect, useState } from "react";
import { Box, Grid, Paper, CircularProgress } from "@mui/material";
import BotControls from "./BotControls";
import ConfigEditor from "./ConfigEditor";
import LogViewer from "./LogViewer";
import DrawdownChart from "./DrawdownChart";
import axios from "axios";
import React from "react";
import { Container, Grid, Paper, Typography } from "@mui/material";
import BotControls from "./BotControls";
import ConfigEditor from "./ConfigEditor";
import LogViewer from "./LogViewer";
import DrawdownChart from "./DrawdownChart";


export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState(null);


  // -----------------------------------------------------------------
  // Load config on mount (needs auth token)
  // -----------------------------------------------------------------
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      // redirect to Okta login (simple example)
      window.location.href = "/login"; // you’ll implement this route in React
      return;
    }

    axios
      .get("/api/config", {
        baseURL: process.env.REACT_APP_API


{/* … inside the Grid layout … */}
<Grid item xs={12} md={6}>
  <Paper sx={{ p: 2 }}>
    <Typography variant="h6" gutterBottom>
      Detailed Equity per Bucket (Grafana)
    </Typography>
    <EmbeddedGrafana
      panelUrl="https://mc.citadel.local/d-solo/abcd1234?orgId=1&panelId=5&from=now-7d&to=now"
    />
  </Paper>
</Grid>

// Inside the Grid layout (e.g., after BotControls)
<Grid item xs={12} md={6}>
  <Paper sx={{ p: 2 }}>
    <Typography variant="h6" gutterBottom>
      Arbitrage Tester
    </Typography>
    <ArbExecutor />
  </Paper>
</Grid>

    export default function Dashboard() {
  return (
    <Container maxWidth="lg" sx={{ my: 4 }}>
      <Typography variant="h4" gutterBottom>
        Citadel Quantum Trader – Admin Console
      </Typography>

      {/* Row 1 – Bot controls + draw‑down chart */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Bot Controls
            </Typography>
            <BotControls />
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Draw‑Down (last 7 days)
            </Typography>
            <DrawdownChart />
          </Paper>
        </Grid>
      </Grid>

      {/* Row 2 – Config editor */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Configuration (YAML)
            </Typography>
            <ConfigEditor />
          </Paper>
        </Grid>
      </Grid>

      {/* Row 3 – Live log viewer */}
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Live Logs (choose a bucket)
            </Typography>
            <LogViewer />
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

