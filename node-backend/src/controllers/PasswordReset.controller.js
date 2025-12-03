import jwt from "jsonwebtoken";
import { Organisation } from "../models/organisation.model.js";
import ApiError from "../utils/apiError.js";
import ApiResponse from "../utils/apiResponse.js";


const resetPassword = async (req, res) => {
  try {
     const otpToken = req.cookies.otpToken;
    const {newPassword,confirmNewPassword} = req.body;
  
   if(!otpToken)
   {
    throw new ApiError(400,"You are not authorized to change password")
   }

     const decodedOtpToken = jwt.verify(otpToken,process.env.OTP_TOKEN_SECRET)

     if(!decodedOtpToken)
     {
      throw new ApiError(400,"Invalid User")
     }
     
    if(newPassword!==confirmNewPassword)
    {
      throw new ApiError(400,"Password and Confirm Password are not same");
    }
  


    const organisation = await Organisation.findById(decodedOtpToken._id).select("+password")


     organisation.password = newPassword;
 
    await organisation.save();
    

    return res.status(200).json(
      new ApiResponse(200,{},"Password successfully reset")
    )
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
};

export {  resetPassword };
