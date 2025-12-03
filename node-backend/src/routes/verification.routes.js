import {Router} from "express"
import { checkOtp, sendOTP } from "../controllers/verification.controller.js";


const router = Router();


router.post("/getOtp/:purpose",sendOTP);
router.post("/verifyOtp",checkOtp)


export default router;