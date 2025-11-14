import {Router} from "express"
import { sendOTP } from "../controllers/verification.controller.js";


const router = Router();


router.post("/getOtp/:purpose",sendOTP);



export default router;