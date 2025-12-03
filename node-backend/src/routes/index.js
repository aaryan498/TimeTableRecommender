import departmentRoutes from "./department.routes.js";
import courseRoutes from "./course.routes.js";
import facultyRoutes from "./faculty.routes.js";
import organisationRoutes from "./organisation.routes.js";
import roomRoutes from "./room.routes.js";
import subjectRoutes from "./subject.routes.js";
import timeslotRoutes from "./timeslot.routes.js";
import timetableRoutes from "./timetable.routes.js";
import verificationRoutes from "./verification.routes.js";
import passwordResetRoutes from "./passwordReset.routes.js";
import chatRoutes from "./chat.routes.js"
import { Router } from "express";

const router = Router();

router.use("/department", departmentRoutes);
router.use("/course", courseRoutes);
router.use("/faculty", facultyRoutes);
router.use("/organisation", organisationRoutes);
router.use("/room", roomRoutes);
router.use("/subject", subjectRoutes);
router.use("/timeslot", timeslotRoutes);
router.use("/timetable", timetableRoutes);
router.use("/verification", verificationRoutes);
router.use("/password-reset", passwordResetRoutes);
router.use("/chatbot",chatRoutes)
export default router;
