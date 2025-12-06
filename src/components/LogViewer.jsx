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

// =====================================================================
// ✅ FIXED: Convert LogEntry class to factory function
// Before: class LogEntry { constructor(content) { ... } }
// After:  function createLogEntry(content) { return { ... } }
// =====================================================================
let logEntryCounter = 0;

function createLogEntry(content) {
  return {
    id: `log-${logEntryCounter++}`,
    content: content,
  };
}

// =====================================================================
// LogViewer Component – displays real-time logs from FastAPI backend
// =====================================================================
export default function LogViewer() {
  const [container, setContainer] = useState("citadel-bot-1");
  const [logLines, setLogLines] = useState([]);
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef(null);

  // ===================================================================
  // Generate container options (6 containers: citadel-bot-1 to citadel-bot-6)
  // ✅ FIXED: Using new Array() instead of [...Array()]
  // ✅ FIXED: Using stable container names as keys instead of array indices
  // ===================================================================
  const containerOptions = new Array(6)
    .fill(null)
    .map((_, i) => ({
      id: `container-${i + 1}`,  // ✅ Stable ID
      label: `citadel-bot-${i + 1}`,
      value: `citadel-bot-${i + 1}`,
    }));

  // ===================================================================
  // When container changes, open a new EventSource (Server‑Sent Events)
  // ===================================================================
  useEffect(() => {
    // Clean up previous stream
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setLoading(true);
    setLogLines([]);

    // FastAPI can stream logs as SSE – we'll use that endpoint
    const es = new EventSource(`/api/logs/${container}`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      // ✅ FIXED: Use factory function instead of class
      const logEntry = createLogEntry(e.data);
      setLogLines((prev) => [...prev, logEntry].slice(-500)); // keep last 500 lines
    };

    es.onerror = (err) => {
      console.error("SSE error", err);
      es.close();
      setLoading(false);
    };

    es.onopen = () => setLoading(false);

    return () => {
      if (es) {
        es.close();
      }
    };
  }, [container]);

  // ===================================================================
  // Render UI
  // ===================================================================
  return (
    <Box sx={{ mt: 2 }}>
      {/* ============================================================= */}
      {/* Container selector dropdown                                   */}
      {/* ============================================================= */}
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel id="container-select-label">Select Bucket</InputLabel>
        <Select
          labelId="container-select-label"
          value={container}
          label="Select Bucket"
          onChange={(e) => setContainer(e.target.value)}
        >
          {/* ✅ FIXED: Using stable container ID as key instead of array index */}
          {containerOptions.map((option) => (
            <MenuItem key={option.id} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* ============================================================= */}
      {/* Log display area                                              */}
      {/* ============================================================= */}
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
          <>
            {/* ✅ FIXED: Using logEntry.id instead of array index */}
            {logLines.map((logEntry) => (
              <Typography key={logEntry.id} variant="body2" component="pre">
                {logEntry.content}
              </Typography>
            ))}
          </>
        )}
      </Paper>
    </Box>
  );
}
