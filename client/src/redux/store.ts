import { configureStore, ThunkAction, Action } from "@reduxjs/toolkit";
import { sideEffectsMiddleware } from "./middleware/sideEffects";
import authApi, { authApi as authApiSlice } from "./reducers/api/auth.api";
import auth from "./reducers/auth";
import documents from "./reducers/documents";
import documentsApi, {
  documentsApi as documentsApiSlice,
} from "./reducers/api/documents.api";
import main from "./reducers/main";
import dialog from "./reducers/dialog";
import notices from "./reducers/notices";
import compositionsApi, {
  compositionsApi as compositionsApiSlice,
} from "./reducers/api/compositions.api";
import compositions from "./reducers/compositions";

export const store = configureStore({
  reducer: {
    authApi,
    main,
    auth,
    documents,
    compositions,
    documentsApi,
    compositionsApi,
    dialog,
    notices,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }).concat([
      authApiSlice.middleware,
      compositionsApiSlice.middleware,
      documentsApiSlice.middleware,
      sideEffectsMiddleware,
    ]),
});

export type AppDispatch = typeof store.dispatch;
export type RootState = ReturnType<typeof store.getState>;
export type AppThunk<ReturnType = void> = ThunkAction<
  ReturnType,
  RootState,
  unknown,
  Action<string>
>;
