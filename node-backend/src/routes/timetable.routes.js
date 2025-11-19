import { Router } from "express";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import { checkGenerationStatus, generateByGivingData, getDetailedTimeTable, getFacultyTimeTables       , getFacultyTimetablesByGroup, getFacultyTimeTablesForSpecific, getInfoPdf, getSectionTimetablesByGroup, getSectionTimeTablesDb, getSingleFacultyTimeTable, getSingleSectionTimeTable, startTimeTableCreation, updateFacultyTimetable } from "../controllers/timetable.controllers.js";
import {   generateAndDownloadAllFacultyTimetables } from "../controllers/facultyTimetable.controllers.js";
import {   updateSectionTimetable } from "../controllers/sectionTimetable.controllers.js";
import { upload } from "../middlewares/multer.middleware.js";
import { saveTimetable } from "../controllers/organisationData.controllers.js";

const router = Router();

// Uncomment to protect routes
router.use(verifyJWT)         

router.post("/generate",startTimeTableCreation);
router.get("/status",checkGenerationStatus);
router.get("/sections",getSectionTimeTablesDb)
router.get("/sections/:section_id",getSingleSectionTimeTable);
router.get("/faculty",getFacultyTimeTables);  
router.post("/upload-pdf",upload.single("file"),getInfoPdf);
// router.get("/faculty/:facultyId/download", downloadFacultyTimetable);
// Section TimeTable 
// router.get("/sections/downloadAll",downloadAllSectionTimetables)

 
router.get("/faculty/downloadAll",generateAndDownloadAllFacultyTimetables);   // in use
router.get("/faculty/:faculty_id",getSingleFacultyTimeTable);
router.get("/detailed",getDetailedTimeTable);




//Routes related to the update of the timetable



router.put("/sectionUpdate",updateSectionTimetable);
router.put("/facultyUpdate",updateFacultyTimetable);


//Saving the manual input for the timetable



router.post("/saveData",saveTimetable)   // i th use


router.post("/sendData",generateByGivingData)
export default router;

router.get("/sections/group",getSectionTimetablesByGroup);
router.get("/faculty/group",getFacultyTimetablesByGroup);
router.get("/faculty/getSpecific",getFacultyTimeTablesForSpecific);