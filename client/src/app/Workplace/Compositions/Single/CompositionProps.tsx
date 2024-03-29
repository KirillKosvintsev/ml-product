import {
  Box,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Typography,
} from "@mui/material";
import { useAppDispatch, useSESelector } from "ducks/hooks";
import { useAllDocumentsQuery } from "ducks/reducers/api/documents.api";
import {
  changeTaskType,
  changeCompositionType,
  changeParamsType,
  changeDataframeId,
  changeTestSize,
} from "ducks/reducers/compositions";
import {
  CompositionType,
  ParamsCompositionType,
  TaskType,
} from "ducks/reducers/types";
import { theme } from "globalStyle/theme";
import { values } from "lodash";
import React from "react";
import { SELECTORS_WIDTH } from "./constants";

export const CompositionProps: React.FC<{ createMode?: boolean }> = ({
  createMode,
}) => {
  const { taskType, compositionType, paramsType, dataframeId, testSize } =
    useSESelector((state) => state.compositions);
  const dispatch = useAppDispatch();
  const { data: allDocuments, isFetching } = useAllDocumentsQuery();

  return (
    <Box>
      <Typography sx={{ mb: theme.spacing(3) }} variant="h5">
        Основное
      </Typography>
      <Box
        sx={{
          display: "flex",
          gap: theme.spacing(1),
          flexWrap: "wrap",
        }}
      >
        <FormControl disabled sx={{ width: SELECTORS_WIDTH }}>
          <InputLabel>Task Type</InputLabel>
          <Select
            disabled={true || !createMode}
            value={taskType}
            label="Task Type"
            onChange={(event) =>
              dispatch(changeTaskType(event.target.value as TaskType))
            }
          >
            {values(TaskType)?.map((x) => (
              <MenuItem key={x} value={x}>
                {x}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl sx={{ width: SELECTORS_WIDTH }}>
          <InputLabel>Composition Type</InputLabel>
          <Select
            disabled={!createMode}
            value={compositionType}
            label="Composition Type"
            onChange={(event) =>
              dispatch(
                changeCompositionType(event.target.value as CompositionType)
              )
            }
          >
            {values(CompositionType)?.map((x) => (
              <MenuItem key={x} value={x}>
                {x}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl sx={{ width: SELECTORS_WIDTH }}>
          <InputLabel>Params Type</InputLabel>
          <Select
            disabled={!createMode}
            value={paramsType}
            label="Params Type"
            onChange={(event) =>
              dispatch(
                changeParamsType(event.target.value as ParamsCompositionType)
              )
            }
          >
            {values(ParamsCompositionType)?.map((x) => (
              <MenuItem key={x} value={x}>
                {x}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl sx={{ width: SELECTORS_WIDTH }}>
          <InputLabel>Document</InputLabel>
          <Select
            disabled={!createMode || isFetching}
            value={allDocuments?.find((x) => x.id === dataframeId)?.filename}
            label="Document"
            onChange={(event) =>
              dispatch(changeDataframeId(event.target.value))
            }
          >
            {allDocuments?.map(({ filename, id }) => (
              <MenuItem key={filename} value={id}>
                {filename}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box sx={{ width: SELECTORS_WIDTH }}>
          <Box sx={{ display: "flex", justifyContent: "space-between" }}>
            <Typography variant="body2">Test Size</Typography>
            <Typography variant="body2">{testSize}</Typography>
          </Box>
          <Slider
            disabled={!createMode}
            value={testSize}
            min={0}
            step={0.01}
            max={1}
            onChange={(_, val) => dispatch(changeTestSize(val as number))}
          />
        </Box>
      </Box>
      <Divider sx={{ mb: theme.spacing(3), mt: theme.spacing(2) }} />
    </Box>
  );
};
