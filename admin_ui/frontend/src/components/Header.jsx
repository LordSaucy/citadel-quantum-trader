import React from "react";
import { AppBar, Toolbar, Typography, Button } from "@mui/material";

export default function Header() {
  const handleLogout = () => {
    localStorage.removeItem("access_token");
    window.location.reload();
  };

  const token = localStorage.getItem("access_token");
  const loggedIn = !!token;

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Citadel Admin Console
        </Typography>
        {loggedIn ? (
          <Button color="inherit" onClick={handleLogout}>
            Logout
          </Button>
        ) : null}
      </Toolbar>
    </AppBar>
  );
}
