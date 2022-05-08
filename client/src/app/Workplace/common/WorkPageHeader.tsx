import { Breadcrumbs, Divider, Typography } from "@mui/material";
import React from "react";
import HomeIcon from "@mui/icons-material/Home";
import { StyledBreadcrumb } from "./styled";
import { pathify } from "ducks/hooks";
import { AppPage, DocumentPage, WorkPage } from "ducks/reducers/types";
import { theme } from "globalStyle/theme";
import { always } from "ramda";
import { useNavigate, useLocation, useParams } from "react-router-dom";
import { HeaderRename } from "../Documents/Single/HeaderRename";

const goHome = (navigate: ReturnType<typeof useNavigate>) => {
  navigate(pathify([AppPage.Workplace, WorkPage.Documents, DocumentPage.List]));
};

const pathItem = (label: () => string, action: () => void) => ({
  label,
  action,
});

const definePathMap = (
  navigate: ReturnType<typeof useNavigate>,
  { docName }: { docName?: string }
) => ({
  [AppPage.Workplace]: pathItem(always("Домой"), () => goHome(navigate)),
  [WorkPage.Documents]: pathItem(always("Документы"), () => goHome(navigate)),
  [DocumentPage.List]: pathItem(always("Все"), () => goHome(navigate)),
  [DocumentPage.Single]: pathItem(
    () => docName || "",
    () => {
      navigate(
        pathify([
          AppPage.Workplace,
          WorkPage.Documents,
          DocumentPage.List,
          docName || "",
        ])
      );
    }
  ),
});

export const WorkPageHeader: React.FC = () => {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { docName } = useParams();
  const PathMap = definePathMap(navigate, { docName });

  const path: (keyof typeof PathMap)[] = pathname
    .split("/")
    .filter((x) => Object.keys(PathMap).includes(x))
    .concat(docName ? DocumentPage.Single : []) as any[];

  return (
    <>
      <Typography sx={{ mb: theme.spacing(2) }} variant="h5">
        {pathname.match(
          pathify([WorkPage.Documents, DocumentPage.List]) + "$"
        ) && "Документы"}
        {docName && <HeaderRename initName={docName} />}
      </Typography>
      <Breadcrumbs sx={{ mb: theme.spacing(2) }}>
        {path.map((x) => (
          <StyledBreadcrumb
            key={x}
            color="secondary"
            label={PathMap[x]?.label()}
            icon={
              x === AppPage.Workplace ? (
                <HomeIcon fontSize="small" />
              ) : undefined
            }
            onClick={PathMap[x]?.action}
          />
        ))}
      </Breadcrumbs>
      <Divider sx={{ mb: theme.spacing(3) }} />
    </>
  );
};