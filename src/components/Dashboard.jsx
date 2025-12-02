import EmbeddedGrafana from "./EmbeddedGrafana";

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
