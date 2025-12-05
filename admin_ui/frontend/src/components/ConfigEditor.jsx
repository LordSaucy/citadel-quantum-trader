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
  // -----------------------------------------------------------------
  // State we actually use
  // -----------------------------------------------------------------
  const [yamlText, setYamlText] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState({
    open: false,
    text: "",
    severity: "info",
  });

  // -----------------------------------------------------------------
  // Load config on mount (convert JSON → YAML for the editor)
  // -----------------------------------------------------------------
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const { data } = await axios.get("/api/config");
        // We only need the YAML representation for the UI
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
  }, []); // ← empty dependency array = run once on mount

  // -----------------------------------------------------------------
  // Save handler – validates YAML before sending it to the backend
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
      // No need to keep a separate config object; the UI already shows the YAML text
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

  // -----------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------
  if (loading) return null; // nothing to show while we fetch

  return (
    <Box sx={{ mt: 2 }}>
      {/* YAML editor */}
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

      {/* Save button */}
      <Button
        variant="contained"
        color="primary"
        onClick={handleSave}
        disabled={saving}
      >
        💾 Save & Reload
      </Button>

      {/* Spinner while saving */}
      {saving && <CircularProgress size={24} sx={{ ml: 2 }} />}

      {/* Snackbar for success / error messages */}
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
