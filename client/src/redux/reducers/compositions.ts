import { createSlice, nanoid, PayloadAction } from "@reduxjs/toolkit";
import {
  CompositionType,
  Model,
  ParamsCompositionType,
  TaskType,
} from "./types";

export type CompositionsSlice = {
  customCompositionName: string;
  models: Record<string, Model>;
  taskType?: TaskType;
  compositionType?: CompositionType;
  paramsType?: ParamsCompositionType;
  documentName: string;
  testSize: number;
};

const initialState: CompositionsSlice = {
  customCompositionName: "",
  models: {},
  documentName: "",
  testSize: 0.2,
};

const compositionsSlice = createSlice({
  name: "compositions",
  initialState,
  reducers: {
    changeCustomCompositionName: (state, action: PayloadAction<string>) =>
      void (state.customCompositionName = action.payload),
    changeTaskType: (state, action: PayloadAction<TaskType>) =>
      void (state.taskType = action.payload),
    changeCompositionType: (state, action: PayloadAction<CompositionType>) =>
      void (state.compositionType = action.payload),
    changeParamsType: (state, action: PayloadAction<ParamsCompositionType>) =>
      void (state.paramsType = action.payload),
    changeDocumentName: (state, action: PayloadAction<string>) =>
      void (state.documentName = action.payload),
    changeTestSize: (state, action: PayloadAction<number>) =>
      void (state.testSize = action.payload),
    addModel: (state) =>
      void (state.models[nanoid()] = { type: null, params: null }),
    changeModel: (state, action: PayloadAction<{ id: string; model: Model }>) =>
      void (state.models[action.payload.id] = action.payload.model),
  },
});

export const {
  changeCustomCompositionName,
  changeTaskType,
  changeCompositionType,
  changeParamsType,
  changeDocumentName,
  changeTestSize,
  addModel,
  changeModel,
} = compositionsSlice.actions;

export default compositionsSlice.reducer;