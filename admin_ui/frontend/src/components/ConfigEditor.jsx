import React, { useEffect, useState } from "react";
import {
  Box,
  Button,
  TextField,
  Snackbar,
  Alert,
  CircularProgress,
} from "@mui/material";
import yaml from "js-yaml";
import axios from "axios";

export default function ConfigEditor() {
  const [configObj, setConfigObj] = useState(null);
  const [yamlText, setYamlText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState({ open: false, text: "", severity: "info" });

  // -----------------------------------------------------------------
  // Load config on mount
  // -----------------------------------------------------------------
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const { data } = await axios.get("/api/config");
        setConfigObj(data);
        setYamlText(yaml.dump(data));
      } catch (e) {
        console.error(e);
        setMsg({
          open: true,
          text: `❌ Failed to load config – ${e.message}`,
          severity: "error",
        });
      } finally {
        setLoading(false);
      }
    };
    fetchConfig();
  }, []);

  // -----------------------------------------------------------------
  // Save handler – validates YAML before sending
  // -----------------------------------------------------------------
  const handleSave = async () => {
    let parsed;
    try {
      parsed = yaml.load(yamlText);
    } catch (e) {
      setMsg({
        open: true,
        text: `❌ Invalid YAML – ${e.message}`,
        severity: "error",
      });
      return;
    }

    setSaving(true);
    try {
      await axios.put("/api/config", parsed);
      setConfigObj(parsed);
      setMsg({
        open: true,
        text: "✅ Config saved and will be hot‑reloaded",
        severity: "success",
      });
    } catch (e) {
      console.error(e);
      setMsg({
        open: true,
        text: `❌ Save failed – ${e.response?.data?.detail || e.message}`,
        severity: "error",
      });
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <CircularProgress />;

  return (
    <Box sx={{ mt: 2 }}>
      <TextField
        label="config.yaml (YAML)"
        multiline
        minRows={20}
        fullWidth
        value={yamlText}
        onChange={(e) => setYamlText(e.target.value)}
        variant="outlined"
        disabled={saving}
        sx={{ fontFamily: "monospace", mb: 2 }}
      />

      <Button
        variant="contained"
        color="primary"
        onClick={handleSave}
        disabled={saving}
      >
        💾 Save & Reload
      </Button>

      {saving && <CircularProgress size={24} sx={{ ml: 2 }} />}

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
