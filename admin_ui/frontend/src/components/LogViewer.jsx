// admin_ui/frontend/src/components/LogViewer.jsx
import React, { useEffect, useState, useRef } from "react";
import {
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Typography,
  CircularProgress,
} from "@mui/material";

/**
 * Helper: generate a stable list of bucket identifiers.
 * If you later add/remove buckets dynamically, replace this
 * static array with a fetch from the backend.
 */
const BUCKETS = Array.from({ length: 6 }, (_, i) => ({
  id: `citadel-bot-${i + 1}`,
  label: `citadel‑bot‑${i + 1}`,
}));

export default function LogViewer() {
  // -----------------------------------------------------------------
  // State
  // -----------------------------------------------------------------
  const [container, setContainer] = useState(BUCKETS[0].id); // default bucket
  const [logLines, setLogLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // -----------------------------------------------------------------
  // Effect – (re)connect to the SSE endpoint whenever `container` changes
  // -----------------------------------------------------------------
  useEffect(() => {
    // Clean up any previous EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setLoading(true);
    setLogLines([]);

    // NOTE: FastAPI streams logs via `/api/logs/:container` (SSE)
    const es = new EventSource(`/api/logs/${container}`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      // Keep only the most recent 500 lines to avoid memory bloat
      setLogLines((prev) => [...prev, e.data].slice(-500));
    };

    es.onerror = (err) => {
      console.error("SSE error", err);
      es.close();
      setLoading(false);
    };

    es.onopen = () => setLoading(false);

    // Cleanup when component unmounts or container changes
    return () => {
      es.close();
    };
  }, [container]);

  // -----------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------
  return (
    <Box sx={{ mt: 2 }}>
      {/* ---------- Bucket selector ---------- */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel id="container-select-label">Select Bucket</InputLabel>
        <Select
          labelId="container-select-label"
          value={container}
          label="Select Bucket"
          onChange={(e) => setContainer(e.target.value as string)}
        >
          {/* Use a stable key derived from the bucket id */}
          {BUCKETS.map((b) => (
            <MenuItem key={b.id} value={b.id}>
              {b.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* ---------- Log pane ---------- */}
      <Paper
        elevation={3}
        sx={{
          p: 2,
          height: 300,
          overflowY: "auto",
          bgcolor: "#111",
          color: "#0f0",
          fontFamily: "monospace",
        }}
      >
        {loading ? (
          <CircularProgress size={24} />
        ) : (
          // Use a stable key – the log line itself is unique enough for a live tail.
          // If you ever get duplicate lines, you can prepend a timestamp or UUID.
          logLines.map((ln, idx) => (
            <Typography key={`${container}-${idx}`} variant="body2" component="pre">
              {ln}
            </Typography>
          ))
        )}
      </Paper>
    </Box>
  );
}
