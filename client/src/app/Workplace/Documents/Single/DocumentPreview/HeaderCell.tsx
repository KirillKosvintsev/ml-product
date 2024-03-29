import React from "react";
import { CategoryMark, DFInfo } from "ducks/reducers/types";
import { theme } from "globalStyle/theme";
import {
  Box,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";
import { StatsGraph } from "./statGraph";
import OpenInFullIcon from "@mui/icons-material/OpenInFull";
import { useAppDispatch } from "ducks/hooks";
import { T } from "ramda";
import { setDialog, setDialogLoading } from "ducks/reducers/dialog";
import { SIMPLE_HEIGHT } from "./contants";
import { OverflowText } from "components/styles";
import { MoreColumnInfo } from "./MoreColumnInfo";
import {
  useDeleteColumnMutation,
  useInfoDocumentQuery,
} from "ducks/reducers/api/documents.api";
import { useParams } from "react-router-dom";
import Filter1Icon from "@mui/icons-material/Filter1";
import TextFormatIcon from "@mui/icons-material/TextFormat";
import DeleteIcon from "@mui/icons-material/Delete";
import { useChangeColumnType } from "./useChangeColumnType";
import CrisisAlertIcon from "@mui/icons-material/CrisisAlert";

const DataHeaderCaption: React.FC<{
  children: React.ReactNode;
  important?: boolean;
}> = (props) => (
  <Typography
    sx={{
      display: "block",
      lineHeight: theme.typography.body1.fontSize,
      color: props.important
        ? theme.palette.info.dark
        : theme.palette.primary.light,
    }}
    variant={props.important ? "body2" : "caption"}
  >
    {props.children}
  </Typography>
);

export const HeaderCell: React.FC<DFInfo & { right?: boolean }> = ({
  type,
  data,
  name,
  not_null_count,
  data_type,
  right,
}) => {
  const { docId } = useParams();
  const dispatch = useAppDispatch();

  const { data: infoDocument } = useInfoDocumentQuery(docId!);
  const [deleteColumn] = useDeleteColumnMutation();

  const setDialogProps = () =>
    data &&
    dispatch(
      setDialog({
        title: `Подробности о ${name}`,
        Content: (
          <MoreColumnInfo
            type={type}
            data={data}
            name={name}
            not_null_count={not_null_count}
            data_type={data_type}
          />
        ),
        onDismiss: T,
        dismissText: "Закрыть",
      })
    );

  const setDialogDeleteProps = () =>
    dispatch(
      setDialog({
        title: `Удаление ${name}`,
        text: `Вы действительно хотите удалить колонку ${name}?`,
        onAccept: async () => {
          dispatch(setDialogLoading(true));
          await deleteColumn({
            dataframe_id: docId!,
            column_name: name,
          });

          dispatch(setDialogLoading(false));
        },
        onDismiss: T,
      })
    );

  const markChange = useChangeColumnType(name);
  const isTarget = name === infoDocument?.column_types?.target;

  return (
    <Box
      sx={{
        flexGrow: 1,
        padding: theme.spacing(1),
        overflow: "hidden",
        textAlign: right ? "right" : "left",
        cursor: "pointer",
        "&:hover": {
          background: theme.palette.info.light,
        },
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <Box
        sx={{
          display: "flex",
          ...OverflowText,
          justifyContent: right ? "flex-end" : "flex-start",
          alignItems: "center",
          color: isTarget ? theme.palette.info.dark : "inherit",
        }}
      >
        {isTarget && (
          <CrisisAlertIcon
            sx={{
              fontSize: theme.typography.body1.fontSize,
              mr: theme.spacing(1),
            }}
          />
        )}
        <Tooltip
          followCursor
          title={`${name}${isTarget ? " (Целевой столбец)" : ""}`}
        >
          <Box sx={{ ...OverflowText }}>{name}</Box>
        </Tooltip>
      </Box>

      <Box
        sx={{
          mb: theme.spacing(1),
          mt: theme.spacing(1),
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <ToggleButton value="size" sx={{ p: "2px" }}>
          <OpenInFullIcon
            onClick={setDialogProps}
            sx={{
              fontSize: theme.typography.caption.fontSize,
            }}
          />
        </ToggleButton>

        <Tooltip
          disableHoverListener={
            !!infoDocument?.column_types?.target && !isTarget
          }
          followCursor
          title={
            !infoDocument?.column_types?.target
              ? "Доступно только после разметки"
              : isTarget
              ? "Смена типа целевого столбца запрещена"
              : ""
          }
        >
          <ToggleButtonGroup
            disabled={!infoDocument?.column_types?.target || isTarget}
            exclusive
            value={type}
            onChange={(_: unknown, val: CategoryMark) => val && markChange(val)}
          >
            <ToggleButton sx={{ p: "2px" }} value={CategoryMark.numeric}>
              <Filter1Icon
                sx={{ fontSize: theme.typography.caption.fontSize }}
              />
            </ToggleButton>
            <ToggleButton sx={{ p: "2px" }} value={CategoryMark.categorical}>
              <TextFormatIcon
                sx={{ fontSize: theme.typography.caption.fontSize }}
              />
            </ToggleButton>
          </ToggleButtonGroup>
        </Tooltip>

        <ToggleButton value="delete" sx={{ p: "2px" }}>
          <DeleteIcon
            onClick={setDialogDeleteProps}
            sx={{
              fontSize: theme.typography.caption.fontSize,
            }}
          />
        </ToggleButton>
      </Box>

      {type && <DataHeaderCaption important>Type: {type}</DataHeaderCaption>}

      <DataHeaderCaption>Not Null: {not_null_count}</DataHeaderCaption>
      <DataHeaderCaption>DataType: {data_type}</DataHeaderCaption>
      {data && (
        <Box sx={{ height: SIMPLE_HEIGHT }}>
          <Box
            sx={{
              display: "flex",
              position: "absolute",
              justifyContent: right ? "flex-end" : "flex-start",
            }}
          >
            <StatsGraph isSimple data={data} type={type} />
          </Box>
        </Box>
      )}
    </Box>
  );
};
