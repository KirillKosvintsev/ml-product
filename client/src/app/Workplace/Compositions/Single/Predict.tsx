import { LoadingButton } from "@mui/lab";
import {
  Box,
  Button,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Pagination,
  Select,
  Tooltip,
  Typography,
} from "@mui/material";
import { OverflowText } from "components/styles";
import { TableFix } from "components/Table";
import { Fixed } from "components/Table/types";
import { useAppDispatch, useSESelector } from "ducks/hooks";
import {
  useAllCompositionsQuery,
  useCompositionInfoQuery,
  usePredictCompositionMutation,
} from "ducks/reducers/api/compositions.api";
import { useAllDocumentsQuery } from "ducks/reducers/api/documents.api";
import { changePredictDataframeId } from "ducks/reducers/compositions";
import { addNotice, SnackBarType } from "ducks/reducers/notices";
import { theme } from "globalStyle/theme";
import { unzip, values, zipObject, keys, entries } from "lodash";
import React, { useState } from "react";
import { SELECTORS_WIDTH } from "./constants";

const convertData = (data: Record<string, (string | number)[]>) =>
  unzip(values(data).map((x) => values(x as any))).map((zipArr) =>
    zipObject(keys(data), zipArr)
  );

const convertToCSV = (arr: any[]) => {
  const array = [Object.keys(arr[0])].concat(arr);

  return array
    .map((it) => {
      return Object.values(it).toString();
    })
    .join("\n");
};

export const Predict: React.FC<{ model_id: string }> = ({ model_id }) => {
  const { predictDataframeId } = useSESelector((state) => state.compositions);
  const [predict, { data }] = usePredictCompositionMutation();
  const [page, setPage] = useState<number>(1);
  const { data: allDocuments, isFetching } = useAllDocumentsQuery();
  const { data: allCompositions } = useAllCompositionsQuery();

  const dispatch = useAppDispatch();

  const convertedData = data ? convertData(data) : [];

  const columns = data
    ? keys(data).map((key) => ({
        accessor: key,
        Header: (
          <Tooltip followCursor title={key}>
            <Box sx={{ ...OverflowText }}>{key}</Box>
          </Tooltip>
        ),
      }))
    : [];

  return (
    <Box>
      <Typography sx={{ mb: theme.spacing(3) }} variant="h5">
        Predict
      </Typography>
      <Box
        sx={{
          display: "flex",
          gap: theme.spacing(2),
          flexWrap: "wrap",
        }}
      >
        <FormControl sx={{ width: SELECTORS_WIDTH }}>
          <InputLabel>Document</InputLabel>
          <Select
            value={
              allDocuments?.find((x) => x.id === predictDataframeId)?.filename
            }
            label="Document"
            onChange={(event) =>
              dispatch(changePredictDataframeId(event.target.value))
            }
          >
            {allDocuments?.map(({ filename, id }) => (
              <MenuItem key={filename} value={id}>
                {filename}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <LoadingButton
          loading={isFetching}
          variant="contained"
          disabled={!predictDataframeId}
          onClick={() =>
            predict({
              model_id,
              dataframe_id: predictDataframeId!,
            }).then(
              (res) =>
                (res as any).error &&
                dispatch(
                  addNotice({
                    label: "Ошибка",
                    type: SnackBarType.error,
                    id: Date.now(),
                  })
                )
            )
          }
          sx={{ width: 200 }}
        >
          Predict
        </LoadingButton>
        {data && (
          <Button
            variant="contained"
            onClick={async () => {
              var element = document.createElement("a");
              element.setAttribute(
                "href",
                "data:text/plain;charset=utf-8," +
                  encodeURIComponent(convertToCSV(convertedData))
              );
              element.setAttribute(
                "download",
                `${allCompositions?.find((x) => x.id === model_id)?.filename}_${
                  allDocuments?.find((x) => x.id === predictDataframeId)
                    ?.filename
                }_predict.csv`
              );

              element.style.display = "none";
              document.body.appendChild(element);

              element.click();

              document.body.removeChild(element);
            }}
          >
            Скачать CSV
          </Button>
        )}
      </Box>

      {data && (
        <Box sx={{ mt: theme.spacing(5) }}>
          <TableFix
            compact
            offHeaderPaddings
            defaultColumnSizing={{ minWidth: 135 }}
            forceResize
            resizable
            data={convertedData.slice((page - 1) * 50, page * 50)}
            columns={columns}
          />
          <Pagination
            sx={{ mt: theme.spacing(2) }}
            page={page}
            onChange={(_, page) => setPage(page)}
            count={Math.ceil(convertedData.length / 50)}
            variant="outlined"
            shape="rounded"
          />
        </Box>
      )}

      <Divider sx={{ mb: theme.spacing(3), mt: theme.spacing(2) }} />
    </Box>
  );
};
