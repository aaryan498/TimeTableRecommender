import { Router } from "express";
import {
  resetPassword,
} from "../controllers/PasswordReset.controller.js";

const router = Router();


router.post("/reset-password", resetPassword);

export default router;
