import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import allRouter from "./routes/index.js";
import { noEmojiMiddleware } from "./middlewares/noEmoji.middleware.js";
import ApiError from "./utils/apiError.js";
import { ZodError } from "zod";

const app = express();

const {FRONTEND_URL} = process.env;

const corsOptions = {
  origin: [FRONTEND_URL,"http://localhost:5173"],
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true,
};

app.use(cors(corsOptions));

app.use(express.json({ limit: "100mb" }));
app.use(express.urlencoded({ extended: true, limit: "100mb" }));


app.use(express.static("public"));


app.use(cookieParser());
app.use(noEmojiMiddleware)
app.use("/api/v1", allRouter);



app.use((err, req, res, next) => {
  console.error(" ERROR OCCURRED:", err);

  let statusCode = err.statusCode || 500;
  let message = err.message || "Internal Server Error";


  if (err instanceof ZodError) {
    statusCode = 400;

    return res.status(statusCode).json({
      success: false,
      message: "Validation failed",
      errors: err.errors.map(e => ({
        path: e.path.join("."),
        message: e.message
      })),
      data: null,
    });
  }


  if (err.name === "ValidationError") {
    statusCode = 400;

    return res.status(statusCode).json({
      success: false,
      message: "Validation error",
      errors: Object.values(err.errors).map(e => e.message),
      data: null,
    });
  }


  if (err.name === "CastError") {
    statusCode = 400;

    return res.status(statusCode).json({
      success: false,
      message: `Invalid ${err.path}: ${err.value}`,
      errors: [],
      data: null,
    });
  }

  if (err instanceof ApiError) {
    return res.status(statusCode).json({
      success: false,
      message: err.message,
      errors: err.errors || [],
      data: err.data || null,
    });
  }


  return res.status(statusCode).json({
    success: false,
    message,
    errors: [],
    data: null,
    stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
  });
});




export default app;
