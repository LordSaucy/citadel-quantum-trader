import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Box, CircularProgress, Snackbar, Alert } from "@mui/material";
import axios from "axios";
import dayjs from "dayjs";

export default function DrawdownChart() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState({ open: false, text: "", severity: "info" });

  const fetchData = async (start, end) => {
    setLoading(true);
    try {
      const { data: resp } = await axios.get("/api/drawdown", {
        params: { start, end },
      });
      // Convert timestamps to JS Date objects for Recharts
      const chartData = resp.points.map((p) => ({
        ts: dayjs(p.ts).format("MMM DD HH:mm"),
        value: p.value,
      }));
      setData(chartData);
    } catch (e) {
      console.error(e);
      setMsg({
        open: true,
        text: `❌ Failed to load draw‑down data – ${e.message}`,
        severity: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  // Load last 7 days on mount
  useEffect(() => {
    const end = new Date().toISOString();
    const start = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    fetchData(start, end);
  }, []);

  return (
    <Box sx={{ mt: 2 }}>
      {loading ? (
        <CircularProgress />
      ) : (
        <LineChart width={800} height={300} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ts" tickFormatter={(t) => t.slice(0, 10)} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#ff7300" dot={false} />
        </LineChart>
      )}

      <Snackbar
        open={msg.open}
        autoHideDuration={6000}
        onClose={() => setMsg({ ...msg, open: false })}
      >
        <Alert severity={msg.severity}>{msg.text}</Alert>
      </Snackbar>
    </Box>
  );
}
