import React, { useEffect, useState } from "react";
import { Card, CardContent, Typography, LinearProgress } from "@mui/material";
import axios from "axios";

export default function StreakPanel({ bucketId }: { bucketId: number }) {
  const [win, setWin] = useState<number>(0);
  const [loss, setLoss] = useState<number>(0);

  useEffect(() => {
    const fetch = async () => {
      const { data } = await axios.get("/api/metrics/streaks", {
        params: { bucket_id: bucketId },
      });
      setWin(data.win);
      setLoss(data.loss);
    };
    fetch();
    const iv = setInterval(fetch, 15_000); // refresh every 15 s
    return () => clearInterval(iv);
  }, [bucketId]);

  const max = Math.max(win, loss, 1);
  const pct = (Math.max(win, loss) / max) * 100;

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle1">
          Bucket {bucketId} – {win > 0 ? "🏆 Win" : "❌ Loss"} streak
        </Typography>
        <Typography variant="h4">{win > 0 ? win : loss}</Typography>
        <LinearProgress
          variant="determinate"
          value={pct}
          color={loss > 0 ? "error" : "success"}
        />
      </CardContent>
    </Card>
  );
}

@app.get("/api/metrics/streaks")
async def get_streaks(bucket_id: int, user=Depends(get_current_user)):
    win = streak_wins.labels(bucket_id=str(bucket_id))._value.get() or 0
    loss = streak_losses.labels(bucket_id=str(bucket_id))._value.get() or 0
    return {"win": int(win), "loss": int(loss)}

