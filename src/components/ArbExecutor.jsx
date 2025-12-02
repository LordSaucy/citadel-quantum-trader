import React, { useState } from "react";
import {
  Box,
  Button,
  TextField,
  Snackbar,
  Alert,
  Typography,
  Paper,
} from "@mui/material";
import axios from "axios";

/**
 * Simple UI to manually fire a triangular arb.
 * In production you would probably hide this behind a feature flag.
 */
export default function ArbExecutor() {
  const [grossProfit, setGrossProfit] = useState(1.0);
  const [status, setStatus] = useState({ open: false, text: "", severity: "info" });
  const [loading, setLoading] = useState(false);

  // Hard‑coded legs for demo – replace with a UI that lets the user pick symbols/volumes
  const legs = [
    { symbol: "EURUSD", side: "buy", volume: 0.01 },
    { symbol: "USDJPY", side: "sell", volume: 0.01 },
    { symbol: "EURJPY", side: "sell", volume: 0.01 },
  ];

  const handleRun = async () => {
    setLoading(true);
    try {
      const payload = { legs, gross_profit_pips: parseFloat(grossProfit) };
      await axios.post("/api/arb/run", payload);
      setStatus({ open: true, text: "✅ Arb executed successfully", severity: "success" });
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setStatus({ open: true, text: `❌ Arb failed – ${msg}`, severity: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        Manual Triangular‑Arb Executor (testing only)
      </Typography>

      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
        <TextField
          label="Gross profit (pips)"
          type="number"
          value={grossProfit}
          onChange={(e) => setGrossProfit(e.target.value)}
          size="small"
          sx={{ width: 180 }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleRun}
          disabled={loading}
        >
          Run Arb
        </Button>
      </Box>

      <Snackbar
        open={status.open}
        autoHideDuration={8000}
        onClose={() => setStatus({ ...status, open: false })}
      >
        <Alert severity={status.severity}>{status.text}</Alert>
      </Snackbar>
    </Paper>
  );
}
