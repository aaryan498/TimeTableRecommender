import { Router } from "express";
import { changePassword, deleteOrganisation, getAllOrganisation, getCurrentOrganisation, getOrganisationFullDetails, loginOrganisation, logoutOrganisation, refreshAccessToken, registerOrganisation, updateAvatar, updateProfile } from "../controllers/organisation.controllers.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import { upload } from "../middlewares/multer.middleware.js";
import { getPreviousSavedData } from "../controllers/organisationData.controllers.js";


const router = Router();


router.post("/register",upload.single("avatar"),registerOrganisation);
router.post("/login",loginOrganisation);



router.post("/verifyLogin",verifyJWT)
router.use(verifyJWT);


router.get("/getAllOrganisations",getAllOrganisation);
router.post("/logout",logoutOrganisation);
router.post("/updateProfile",updateProfile);
router.post("/changePassword",changePassword);
router.post("/update-avatar",upload.single("avatar"),updateAvatar)

router.delete("/delete",deleteOrganisation);
router.get("/refresh-token",refreshAccessToken);


router.get("/getCurrentOrganisation",getCurrentOrganisation);


// router.get("getOrganisationFullDetails",getOrganisationFullDetails)
router.get("/getOrganisationSavedData",verifyJWT,getPreviousSavedData);

export default router;