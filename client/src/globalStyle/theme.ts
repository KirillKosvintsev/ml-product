import { createTheme, ThemeOptions } from "@mui/material/styles";

const themeOptions: ThemeOptions = {
  palette: {
    primary: {
      main: "#24292f",
    },
    secondary: {
      main: "#f6f8fa",
    },
  },
  typography: {
    fontSize: 24,
    fontFamily: "Jost",
    htmlFontSize: 24,
  },
};

export const theme = createTheme(themeOptions);