import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { ROUTES } from "../../constants";
import { addAuthHeader } from "./helpers";
import {
  ColumnMarksPayload,
  ColumnStats,
  DescribeDoc,
  DocumentInfo,
  DocumentInfoShort,
  DocumentMethod,
  DocumentStatsInfo,
  FullDocument,
  TaskType,
} from "../types";
import { values } from "lodash";

const buildFileForm = (file: File) => {
  const form = new FormData();
  form.append("file", file);

  return form;
};

enum Tags {
  documents = "documents",
  singleDocument = "singleDocument",
  columnMarks = "columnMarks",
  pipeline = "pipeline",
}

export const documentsApi = createApi({
  reducerPath: "documentsApi",
  baseQuery: fetchBaseQuery({
    baseUrl: ROUTES.DOCUMENTS.BASE,
    prepareHeaders: addAuthHeader,
  }),
  tagTypes: Object.values(Tags),
  endpoints: (builder) => ({
    document: builder.query<FullDocument, { filename: string; page: number }>({
      query: ({ filename, page }) => ({
        url: ROUTES.DOCUMENTS.SHOW,
        params: { filename, page },
      }),
      providesTags: [Tags.singleDocument],
    }),
    postDocument: builder.mutation<string, { filename: string; file: File }>({
      query: ({ filename, file }) => ({
        url: ROUTES.DOCUMENTS.BASE,
        params: { filename },
        method: "POST",
        body: buildFileForm(file),
      }),
      invalidatesTags: [Tags.documents],
    }),
    deleteDocument: builder.mutation<string, string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.BASE,
        params: { filename },
        method: "DELETE",
      }),
      invalidatesTags: [Tags.documents],
    }),
    infoDocument: builder.query<DocumentInfo, string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.INFO,
        params: { filename },
      }),
      providesTags: [Tags.singleDocument],
    }),
    infoStatsDocument: builder.query<DocumentStatsInfo, string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.STATS_INFO,
        params: { filename },
      }),
      providesTags: [Tags.singleDocument],
      transformResponse: (res: Record<string, Record<string, string>>) =>
        Object.fromEntries(
          Object.entries(res).map(([key, val]) => [key, values(val)])
        ) as DocumentStatsInfo,
    }),
    describeDocument: builder.query<DescribeDoc, string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.DESCRIBE,
        params: { filename },
      }),
      providesTags: [Tags.singleDocument],
    }),
    pipelineDocument: builder.query<
      string,
      { document_from: string; document_to: string }
    >({
      query: ({ document_from, document_to }) => ({
        url: ROUTES.DOCUMENTS.PIPE,
        params: { document_from, document_to },
      }),
      providesTags: [Tags.pipeline],
    }),
    allDocuments: builder.query<DocumentInfoShort[], void>({
      query: () => ({
        url: ROUTES.DOCUMENTS.ALL,
      }),
      providesTags: [Tags.documents],
    }),
    downloadDocument: builder.mutation<null, string>({
      async queryFn(filename) {
        const res = await fetch(
          `${ROUTES.DOCUMENTS.BASE}${ROUTES.DOCUMENTS.DOWNLOAD}?filename=${filename}`,
          {
            headers: { Authorization: `Bearer ${localStorage.authToken}` },
          }
        );

        const blob = await res.blob();

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        return {
          data: null,
          meta: null,
        };
      },
    }),
    renameDocument: builder.mutation<
      string,
      { filename: string; new_filename: string }
    >({
      query: ({ filename, new_filename }) => ({
        url: ROUTES.DOCUMENTS.RENAME,
        params: { filename, new_filename },
        method: "PUT",
      }),
      invalidatesTags: [Tags.documents],
    }),
    columnsDocument: builder.query<string[], string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.COLUMNS,
        params: { filename },
      }),
    }),

    columnMarksDocument: builder.query<ColumnMarksPayload, string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.COLUMN_MARKS,
        params: { filename },
      }),
      providesTags: [Tags.singleDocument],
    }),
    changeColumnMarks: builder.mutation<
      string,
      { filename: string; body: ColumnMarksPayload }
    >({
      query: ({ filename, body }) => ({
        url: ROUTES.DOCUMENTS.COLUMN_MARKS,
        params: { filename },
        body,
        method: "PUT",
      }),
      invalidatesTags: [Tags.singleDocument],
    }),
    applyDocMethod: builder.mutation<
      string,
      { filename: string; function_name: DocumentMethod }
    >({
      query: ({ filename, function_name }) => ({
        url: ROUTES.DOCUMENTS.APPLY_METHOD,
        params: { filename, function_name },
        method: "POST",
      }),
      invalidatesTags: [Tags.pipeline, Tags.singleDocument],
    }),
    columnsStatsDocument: builder.query<ColumnStats[], string>({
      query: (filename) => ({
        url: ROUTES.DOCUMENTS.STATS_COLUMNS,
        params: { filename },
      }),
      providesTags: [Tags.singleDocument],
    }),
    selectDocumentTarget: builder.mutation<
      ColumnStats[],
      { filename: string; targetColumn: string; taskType: TaskType }
    >({
      query: ({ targetColumn, filename, taskType }) => ({
        url: ROUTES.DOCUMENTS.SELECT_TARGET,
        params: { filename, target_column: targetColumn, task_type: taskType },
        method: "PUT",
      }),
      invalidatesTags: [Tags.singleDocument],
    }),
  }),
});

export const {
  useAllDocumentsQuery,
  usePostDocumentMutation,
  useDeleteDocumentMutation,
  useInfoDocumentQuery,
  useRenameDocumentMutation,
  useDownloadDocumentMutation,
  useColumnMarksDocumentQuery,
  useDocumentQuery,
  useColumnsDocumentQuery,
  useChangeColumnMarksMutation,
  useApplyDocMethodMutation,
  useDescribeDocumentQuery,
  useInfoStatsDocumentQuery,
  useColumnsStatsDocumentQuery,
  useSelectDocumentTargetMutation,
} = documentsApi;
export default documentsApi.reducer;
