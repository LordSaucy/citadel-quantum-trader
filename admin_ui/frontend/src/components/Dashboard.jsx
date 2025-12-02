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
