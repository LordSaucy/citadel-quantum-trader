import React, { useState } from "react";
import {
  Button,
  Stack,
  Snackbar,
  Alert,
  CircularProgress,
} from "@mui/material";
import axios from "axios";

export default function BotControls() {
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState({ open: false, text: "", severity: "info" });

  const callApi = async (endpoint, successMsg) => {
    setLoading(true);
    try {
      await axios.post(`/api/${endpoint}`);
      setMsg({ open: true, text: successMsg, severity: "success" });
    } catch (e) {
      console.error(e);
      setMsg({
        open: true,
        text: `❌ ${endpoint} failed – ${e.response?.data?.detail || e.message}`,
        severity: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Stack direction="row" spacing={2}>
        <Button
          variant="contained"
          color="primary"
          disabled={loading}
          onClick={() => callApi("bot/resume", "All buckets resumed")}
        >
          ▶️ Resume
        </Button>

        <Button
          variant="contained"
          color="warning"
          disabled={loading}
          onClick={() => callApi("bot/pause", "All buckets paused")}
        >
          ⏸️ Pause
        </Button>

        <Button
          variant="contained"
          color="error"
          disabled={loading}
          onClick={() =>
            callApi("bot/kill", "Kill‑switch activated – trading stopped")
          }
        >
          🛑 Kill‑Switch
        </Button>
      </Stack>

      {/* Loading spinner overlay */}
      {loading && <CircularProgress size={24} sx={{ ml: 2 }} />}

      {/* Snackbar for feedback */}
      <Snackbar
        open={msg.open}
        autoHideDuration={6000}
        onClose={() => setMsg({ ...msg, open: false })}
      >
        <Alert severity={msg.severity}>{msg.text}</Alert>
      </Snackbar>
    </>
  );
}
