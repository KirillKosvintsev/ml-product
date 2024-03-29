import { Box, CssBaseline, ThemeProvider } from "@mui/material";
import { Authentication } from "./app/Authentication";
import { browserHistory, Matcher, pathify, useSESelector } from "ducks/hooks";
import { AppPage } from "ducks/reducers/types";
import React, { useEffect, useLayoutEffect, useState } from "react";
import { theme } from "./globalStyle/theme";
import { CenteredContainer } from "components/muiOverride";
import { Circles } from "react-loader-spinner";
import { withOpacity } from "./globalStyle/theme";
import { Workplace } from "./app/Workplace";
import { DialogCustom } from "components/Dialog";
import { Route, Routes } from "react-router";
import { Router } from "react-router-dom";
import { Notices } from "components/Notice";
import { useCentrifugoSocketQuery } from "ducks/reducers/api/auth.api";
import { socketManager } from "ducks/reducers/api/socket";

const App: React.FC = () => {
  const { data: socketToken } = useCentrifugoSocketQuery(undefined, {
    skip: !localStorage.authToken,
  });

  useEffect(() => {
    if (!socketManager.isCreated && socketToken)
      socketManager.createSocket(socketToken);
  }, [socketToken]);

  const { isBlockingLoader } = useSESelector((state) => state.main);
  const [state, setState] = useState({
    action: browserHistory.action,
    location: browserHistory.location,
  });

  useLayoutEffect(() => browserHistory.listen(setState), []);
  return (
    <Router location={state.location} navigator={browserHistory}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <DialogCustom />
        <Notices />
        <Box sx={{ backgroundColor: "secondary.main" }}>
          {isBlockingLoader && (
            <CenteredContainer
              sx={{
                position: "fixed",
                height: "100vh",
                maxWidth: "100vw !important",
                zIndex: 100,
                bgcolor: (theme) =>
                  withOpacity(theme.palette.primary.main, 0.98),
              }}
            >
              <Circles width={100} color={theme.palette.secondary.dark} />
            </CenteredContainer>
          )}

          <Routes>
            <Route
              path={pathify([AppPage.Authentication], {
                matcher: Matcher.start,
              })}
              element={<Authentication />}
            />
            <Route
              path={pathify([AppPage.Workplace], {
                matcher: Matcher.start,
              })}
              element={<Workplace />}
            />
            <Route path="/" element={<Authentication />} />
          </Routes>
        </Box>
      </ThemeProvider>
    </Router>
  );
};

export default App;
