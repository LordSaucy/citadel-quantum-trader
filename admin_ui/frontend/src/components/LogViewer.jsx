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

export default function LogViewer() {
  const [container, setContainer] = useState("citadel-bot-1"); // default bucket
  const [logLines, setLogLines] = useState([]);
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef(null);

  // -----------------------------------------------------------------
  // When container changes, open a new EventSource (Server‑Sent Events)
  // -----------------------------------------------------------------
  useEffect(() => {
    // Clean up previous stream
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setLoading(true);
    setLogLines([]);

    // FastAPI can stream logs as SSE – we’ll use that endpoint
    const es = new EventSource(`/api/logs/${container}`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      setLogLines((prev) => [...prev, e.data].slice(-500)); // keep last 500 lines
    };
    es.onerror = (err) => {
      console.error("SSE error", err);
      es.close();
      setLoading(false);
    };
    es.onopen = () => setLoading(false);

    return () => es.close();
  }, [container]);

  // -----------------------------------------------------------------
  // Render UI
  // -----------------------------------------------------------------
  return (
    <Box sx={{ mt: 2 }}>
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel id="container-select-label">Select Bucket</InputLabel>
        <Select
          labelId="container-select-label"
          value={container}
          label="Select Bucket"
          onChange={(e) => setContainer(e.target.value)}
        >
          {/* List of bucket containers – you can generate this list from the
              backend if you prefer; hard‑coded for demo */}
          {[...Array(6)].map((_, i) => (
            <MenuItem key={i} value={`citadel-bot-${i + 1}`}>
              citadel‑bot‑{i + 1}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

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
          logLines.map((ln, idx) => (
            <Typography key={idx} variant="body2" component="pre">
              {ln}
            </Typography>
          ))
        )}
      </Paper>
    </Box>
  );
}
