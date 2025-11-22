import {Router} from "express"
import { chatBot } from "../controllers/chatbot.controller.js";








const router = Router()







router.post("/chat",chatBot)

export default router;