import React from "react";
import { Box } from "@mui/material";

export default function EmbeddedGrafana({ panelUrl }) {
  // Example panelUrl:
  // https://mc.citadel.local/d-solo/abcd1234?orgId=1&panelId=2&from=now-7d&to=now
  return (
    <Box sx={{ border: "1px solid #444", borderRadius: 1, overflow: "hidden" }}>
      <iframe
        src={panelUrl}
        width="100%"
        height="400"
        frameBorder="0"
        sandbox="allow-scripts allow-same-origin"
        title="Grafana panel"
      />
    </Box>
  );
}
