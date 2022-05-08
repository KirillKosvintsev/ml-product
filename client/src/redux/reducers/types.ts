export enum AuthPage {
  auth = "auth",
  register = "register",
}

export enum AppPage {
  Authentication = "authentication",
  Workplace = "workplace",
}

export enum WorkPage {
  Documents = "documents",
}

export enum DocumentPage {
  List = "list",
  Single = "single",
}

export type AuthSlice = {
  passwordInput: string;
  secondPasswordInput: string;
  emailInput: string;
};

export type MainSlice = {
  isBlockingLoader: boolean;
};

export type DecodedToken = {
  user_id: string;
  exp: number;
};

export type AuthPayload = { username: string; password: string };

export type RegisterPayload = {};

export type EmittedToken = {
  access_token: string;
  token_type: string;
};

declare global {
  interface Window {
    localStorage: { authToken: string };
  }
}

export type Document = {
  [key: string]: string[] | number[];
};

export type DocumentInfo = DocumentInfoShort & {
  id: string;
  pipeline: string[];
  column_marks: {
    numeric: string[];
    categorical: string[];
    target: string;
  };
};

export type DocumentInfoShort = {
  name: string;
  upload_date: string;
  change_date: string;
};

export type ColumnMarksPayload = {
  numeric: string[];
  categorical: string[];
  target: string;
};