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

