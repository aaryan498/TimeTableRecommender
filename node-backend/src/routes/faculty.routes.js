import { Router } from "express";
import { createFaculty, deleteFaculty, forgotPassword, getAllFaculty, getFacultyProfile, loginFaculty, requestTimetableUpdate, resetPassword } from "../controllers/faculty.controllers.js";

import { verifyJWT } from "../middlewares/auth.middleware.js";

const router = Router();


router.use(verifyJWT)
router.post("/createFaculty",createFaculty);
router.get("/getAllFaculties",getAllFaculty);
router.post("/loginFaculty",loginFaculty);
router.get("/getFacultyProfile",getFacultyProfile);
router.delete("/deleteFaculty/:facId",deleteFaculty)
router.post("/reqTimetableUpdate",requestTimetableUpdate);
router.post("/forgotPassword",forgotPassword);
router.post("/resetPassword",resetPassword);

// router.get("/saveFacultyTimeTables",saveFacultyTimetables)

export default router;