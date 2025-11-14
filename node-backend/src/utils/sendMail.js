import nodemailer from "nodemailer"
import dotenv from "dotenv"

// dotenv.config({
//   path:"../../.env"
// })

const transporter = nodemailer.createTransport({
  secure: true,
  host: "smtp.gmail.com",
  port: 465,
  service: "gmail",
  auth: {
    user: process.env.COMP_EMAIL,
    pass: process.env.COMP_PASS,
  },
});

const sendEmail = (email,subject,data,purpose)=>{transporter.sendMail({
 from:`"TimeTableScheduler.com" ${""}`,
 to:email,
 subject:subject,
 html:` 
    <h1>TimeTableScheduler.com</h1>
    <h2>Your OTP for ${purpose} :<h2>
 
    <h3>    ${data}     </h3>`




})
         
};

export {sendEmail}