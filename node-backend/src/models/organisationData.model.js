import mongoose, { Schema } from 'mongoose';

const organisationData = new mongoose.Schema({
  organisationId: {
    type: Schema.Types.ObjectId,
    ref: "Organisation"
  },
  course:{
    type:String,
    lowercase:true,
    trim:true,
    required:true
  },
year:{
      type:String,
      required:true,
      trim:true,
      lowercase:true
    },

  college_info: {
    name: { type: String, required: true, minlength: 2, maxlength: 100 },
    session: { type: String, required: true, minlength: 2, maxlength: 20 },
    effective_date: { type: Date, required: true }
  },

  time_slots: {
    periods: [
      {
        id: Number,
        start_time: { type: String, minlength: 1, maxlength: 20 },
        end_time: { type: String, minlength: 1, maxlength: 20 }
      }
    ],
    working_days: [{ type: String, minlength: 3, maxlength: 15 }],
    break_periods: [Number],
    lunch_period: Number,
    mentorship_period: Number
  },

  departments: [
    {
      dept_id: { type: String, minlength: 1, maxlength: 50 },
      name: { type: String, minlength: 2, maxlength: 100 },
      sections: [
        {
          section_id: { type: String, minlength: 1, maxlength: 50 },
          name: { type: String, minlength: 1, maxlength: 50 },
          semester: { type: String, minlength: 1, maxlength: 20 },
          year: { type: String, minlength: 1, maxlength: 20 },
          room: { type: String, minlength: 1, maxlength: 20 },
          student_count: Number,
          coordinator: { type: String, minlength: 2, maxlength: 100 }
        }
      ]
    }
  ],

  subjects: { type: Array },
  labs: { type: Array },
  faculty: { type: Array },
  rooms: { type: Array },

  constraints: {
    hard_constraints: Object,
    soft_constraints: Object
  },

  special_requirements: Object,
  genetic_algorithm_params: Object

}, { timestamps: true });

export const OrganisationData = mongoose.model('OrganisationData', organisationData);
