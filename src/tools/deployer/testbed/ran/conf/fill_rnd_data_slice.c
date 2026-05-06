/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */



#include "fill_rnd_data_slice.h"
#include "../../src/util/time_now_us.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// ====================================================
// 1. GLOBAL STORAGE (The Singleton)
// ====================================================
static bool init_flag = false;
static slice_ind_data_t slice_data_global;

// ====================================================
// 2. HELPER FUNCTIONS: DEEP COPY
//    (Required because the data is nested with pointers)
// ====================================================

static char* my_strdup(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src) + 1;
    char* dst = malloc(len);
    if (dst) memcpy(dst, src, len);
    return dst;
}

static void deep_copy_edf(edf_slice_t* dst, const edf_slice_t* src) {
    // Copy scalars
    *dst = *src;
    // Copy dynamic array 'over'
    if (src->len_over > 0 && src->over) {
        dst->over = calloc(src->len_over, sizeof(uint32_t));
        assert(dst->over != NULL);
        memcpy(dst->over, src->over, src->len_over * sizeof(uint32_t));
    } else {
        dst->over = NULL;
        dst->len_over = 0;
    }
}

static void deep_copy_params(slice_params_t* dst, const slice_params_t* src) {
    dst->type = src->type;
    // Copy the union
    // Most types (Static, NVS, SCN19) are flat structs (no pointers inside), 
    // so we can copy them directly.
    // EDF has a pointer, so it needs special handling.
    if (src->type == SLICE_ALG_SM_V0_EDF) {
        deep_copy_edf(&dst->u.edf, &src->u.edf);
    } else {
        dst->u = src->u; 
    }
}

static void deep_copy_ul_dl(ul_dl_slice_conf_t* dst, const ul_dl_slice_conf_t* src) {
    // 1. Copy Scheduler Name
    dst->len_sched_name = src->len_sched_name;
    dst->sched_name = my_strdup(src->sched_name);

    // 2. Copy Slices Array
    dst->len_slices = src->len_slices;
    if (src->len_slices > 0 && src->slices) {
        dst->slices = calloc(dst->len_slices, sizeof(fr_slice_t));
        assert(dst->slices != NULL);

        for (uint32_t i = 0; i < dst->len_slices; ++i) {
            dst->slices[i].id = src->slices[i].id;
            
            // Strings
            dst->slices[i].len_label = src->slices[i].len_label;
            dst->slices[i].label = my_strdup(src->slices[i].label);
            
            dst->slices[i].len_sched = src->slices[i].len_sched;
            dst->slices[i].sched = my_strdup(src->slices[i].sched);

            // Params (Recursive copy)
            deep_copy_params(&dst->slices[i].params, &src->slices[i].params);
        }
    } else {
        dst->slices = NULL;
    }
}

static void deep_copy_ue_conf(ue_slice_conf_t* dst, const ue_slice_conf_t* src) {
    dst->len_ue_slice = src->len_ue_slice;
    if (src->len_ue_slice > 0 && src->ues) {
        dst->ues = calloc(dst->len_ue_slice, sizeof(ue_slice_assoc_t));
        assert(dst->ues != NULL);
        // Assoc struct has no pointers, simple memcpy is fine
        memcpy(dst->ues, src->ues, dst->len_ue_slice * sizeof(ue_slice_assoc_t));
    } else {
        dst->ues = NULL;
    }
}

// Function to copy the GLOBAL data into the USER provided pointer
static void copy_global_to_user(slice_ind_data_t* dst) {
    assert(dst != NULL);

    // 1. Copy Header (Simple struct)
    dst->hdr = slice_data_global.hdr;

    // 2. Copy Proc ID (Pointer allocation)
    if (slice_data_global.proc_id) {
        dst->proc_id = calloc(1, sizeof(slice_call_proc_id_t));
        *dst->proc_id = *slice_data_global.proc_id;
    }

    // 3. Copy Message Content
    dst->msg.tstamp = time_now_us(); // Always update timestamp to 'now'
    
    // Deep Copy UL/DL Lists
    deep_copy_ul_dl(&dst->msg.slice_conf.ul, &slice_data_global.msg.slice_conf.ul);
    deep_copy_ul_dl(&dst->msg.slice_conf.dl, &slice_data_global.msg.slice_conf.dl);
    
    // Deep Copy UE List
    deep_copy_ue_conf(&dst->msg.ue_slice_conf, &slice_data_global.msg.ue_slice_conf);
}

// ====================================================
// 3. GENERATOR FUNCTIONS (Standard Random Logic)
//    These simply fill whatever pointer is given to them.
// ====================================================

static void fill_static_slice(static_slice_t* sta)
{
  assert(sta != NULL);
  sta->pos_high = abs(rand()%25);
  sta->pos_low = abs(rand()%25);
}

static void fill_nvs_slice(nvs_slice_t* nvs)
{
  assert(nvs != NULL);
  const uint32_t type = abs(rand() % SLICE_SM_NVS_V0_END);
  if(type == SLICE_SM_NVS_V0_RATE ){
    nvs->conf = SLICE_SM_NVS_V0_RATE; 
    nvs->u.rate.u2.mbps_reference = 0.8; 
    nvs->u.rate.u1.mbps_required = 10.0;
  } else if(type ==SLICE_SM_NVS_V0_CAPACITY ){
    nvs->conf = SLICE_SM_NVS_V0_CAPACITY; 
    nvs->u.capacity.u.pct_reserved = 15.0;
  } else {
    assert(0!=0 && "Unknown type");
  }
}

static void fill_scn19_slice(scn19_slice_t* scn19)
{
  assert(scn19 != NULL);
  const uint32_t type = abs(rand()% SLICE_SCN19_SM_V0_END);
  if(type == SLICE_SCN19_SM_V0_DYNAMIC ){
    scn19->conf = SLICE_SCN19_SM_V0_DYNAMIC ;
    scn19->u.dynamic.u2.mbps_reference = 10.0 * fabs((float)rand()/(float)RAND_MAX); 
    scn19->u.dynamic.u1.mbps_required = 8.0 * fabs((float)rand()/(float)RAND_MAX); 
  } else if(type == SLICE_SCN19_SM_V0_FIXED ) {
    scn19->conf = SLICE_SCN19_SM_V0_FIXED; 
    scn19->u.fixed.pos_high = abs(rand()%14);
    scn19->u.fixed.pos_low = abs(rand()%10);
  } else if(type ==SLICE_SCN19_SM_V0_ON_DEMAND){
    scn19->conf = SLICE_SCN19_SM_V0_ON_DEMAND;
    scn19->u.on_demand.log_delta = 1.0 * fabs((float)rand()/(float)RAND_MAX);
    scn19->u.on_demand.tau = abs(rand()%256);
    scn19->u.on_demand.pct_reserved = fabs((float)rand()/(float)RAND_MAX);
  } else {
    assert(0 != 0 && "Unknown type!!");
  }
}

static void fill_edf_slice(edf_slice_t* edf)
{
  assert(edf != NULL);
  int mod = 32;
  edf->deadline = abs(rand()%mod);
  edf->guaranteed_prbs = abs(rand()%mod);
  edf->max_replenish = abs(rand()%mod);
  edf->len_over = 1; 

  if(edf->len_over > 0){
    edf->over = calloc(edf->len_over, sizeof(uint32_t));
    assert(edf->over != NULL && "Memory exhausted");
  }
  for(uint32_t i = 0; i < edf->len_over; ++i){
    edf->over[i] = abs(rand()%mod);
  }
}

static void fill_ul_dl_slice(ul_dl_slice_conf_t* slice)
{
  assert(slice != NULL);

  char const* name = "MY SLICE";
  slice->len_sched_name = strlen(name);
  slice->sched_name = malloc(strlen(name) + 1);
  strcpy(slice->sched_name, name);

  slice->len_slices = abs(rand()%4) + 1; // Ensure at least 1 for testing

  if(slice->len_slices > 0){
    slice->slices = calloc(slice->len_slices, sizeof(fr_slice_t));
    assert(slice->slices != NULL && "memory exhausted");
  }

  for(uint32_t i = 0; i < slice->len_slices; ++i){
    slice->slices[i].id = abs(rand()%1024);
    fr_slice_t* s = &slice->slices[i];

    const char* label = "This is my label";
    s->len_label = strlen(label);
    s->label = malloc(s->len_label + 1);
    strcpy(s->label, label);

    const char* sched_str = "Scheduler string";
    s->len_sched = strlen(sched_str); 
    s->sched = malloc(s->len_sched + 1);
    strcpy(s->sched, sched_str);

    uint32_t type = abs(rand()% SLICE_ALG_SM_V0_END);

    if(type == SLICE_ALG_SM_V0_NONE || type == SLICE_ALG_SM_V0_SCN19)
      type = SLICE_ALG_SM_V0_STATIC; 

    if(type == SLICE_ALG_SM_V0_NONE ){
      s->params.type =SLICE_ALG_SM_V0_NONE; 
    } else if (type == SLICE_ALG_SM_V0_STATIC ){
      s->params.type = SLICE_ALG_SM_V0_STATIC; 
      fill_static_slice(&s->params.u.sta);
    } else if (type == SLICE_ALG_SM_V0_NVS){
      s->params.type =  SLICE_ALG_SM_V0_NVS; 
      fill_nvs_slice(&s->params.u.nvs);
    } else if (type == SLICE_ALG_SM_V0_SCN19) {
      s->params.type = SLICE_ALG_SM_V0_SCN19; 
      fill_scn19_slice(&s->params.u.scn19);
    } else if (type == SLICE_ALG_SM_V0_EDF){
      s->params.type =  SLICE_ALG_SM_V0_EDF; 
      fill_edf_slice(&s->params.u.edf);
    }
  }
}

static void fill_slice_conf(slice_conf_t* conf)
{
  assert(conf != NULL);
  fill_ul_dl_slice(&conf->ul);
  fill_ul_dl_slice(&conf->dl);
}

static void fill_ue_slice_conf(ue_slice_conf_t* conf)
{
  assert(conf != NULL);
  conf->len_ue_slice = abs(rand()%10);
  if(conf->len_ue_slice > 0){
    conf->ues = calloc(conf->len_ue_slice, sizeof(ue_slice_assoc_t));
    assert(conf->ues != NULL && "memory exhausted");
  }
  for(uint32_t i = 0; i < conf->len_ue_slice; ++i){
    conf->ues[i].rnti = abs(rand()%1024);  
    conf->ues[i].dl_id = abs(rand()%16); 
    conf->ues[i].ul_id = abs(rand()%16); 
  }
}

// ====================================================
// 4. MAIN ENTRY POINT (SINGLETON LOGIC)
// ====================================================

void fill_slice_ind_data(slice_ind_data_t* ind_msg)
{
  assert(ind_msg != NULL);

  // STEP A: If it's the first run, generate data into GLOBAL storage
  if (init_flag == false) {
      printf("[SLICE_SIM] Generating Initial Random Data (Singleton)...\n");
      srand(time(0));

      // 1. Clear Global Memory
      memset(&slice_data_global, 0, sizeof(slice_ind_data_t));

      // 2. Allocate Proc ID
      slice_data_global.proc_id = calloc(1, sizeof(slice_call_proc_id_t));

      // 3. Generate Configs directly into Global variables
      // Note: We pass the pointers to the global struct parts here
      fill_slice_conf(&slice_data_global.msg.slice_conf);
      fill_ue_slice_conf(&slice_data_global.msg.ue_slice_conf);

      // 4. Mark as Initialized
      init_flag = true;
  } else {
      //printf("[SLICE_SIM] Using Cached Global Data...\n");
  }

  // STEP B: Always DEEP COPY the global data to the user's buffer
  // This ensures the user gets their own memory to modify/free, 
  // while the global storage remains untouched.
  copy_global_to_user(ind_msg);
}

// =========================================================
// HELPERS: MODIFY GLOBAL STATE
// =========================================================

// =========================================================
// HELPER: Add OR Modify Slices (Upsert)
// =========================================================
static void global_upsert_slices(ul_dl_slice_conf_t* global_list, ul_dl_slice_conf_t* req_data)
{
    // Iterate through every slice provided in the Request
    for (uint32_t i = 0; i < req_data->len_slices; i++) {
        fr_slice_t* req_s = &req_data->slices[i]; // The incoming data
        fr_slice_t* target = NULL;

        // Step 1: Search for existing ID in Global List
        for(uint32_t j = 0; j < global_list->len_slices; j++) {
            if(global_list->slices[j].id == req_s->id) {
                target = &global_list->slices[j]; // Found it!
                printf("[E2 Agent] ID %d exists. Modifying...\n", req_s->id);
                break;
            }
        }

        // Step 2: Prepare the Target Memory
        if (target != NULL) {
            // --- CASE A: MODIFY EXISTING ---
            // Free the OLD string memory to avoid leaks before overwriting
            if(target->label) free(target->label);
            if(target->sched) free(target->sched);
            // Free EDF specific array if it exists
            if(target->params.type == SLICE_ALG_SM_V0_EDF && target->params.u.edf.over) {
                 free(target->params.u.edf.over);
            }
            // (Target pointer is already valid, pointing to existing slot)
        } 
        else {
            // --- CASE B: ADD NEW ---
            printf("[E2 Agent] ID %d is new. Adding...\n", req_s->id);
            
            // Expand the array by 1
            global_list->len_slices++;
            global_list->slices = realloc(global_list->slices, global_list->len_slices * sizeof(fr_slice_t));
            assert(global_list->slices != NULL && "Memory exhausted");

            // Point target to the new last slot
            target = &global_list->slices[global_list->len_slices - 1];
            
            // Set the ID (only needed for new items)
            target->id = req_s->id;
        }

        // Step 3: Deep Copy Data from Request to Target (Global)
        // (This logic works for both new and modified slices)
        
        target->len_label = req_s->len_label;
        target->label = my_strdup(req_s->label); // Copy string from request

        target->len_sched = req_s->len_sched;
        target->sched = my_strdup(req_s->sched); // Copy string from request

        // Deep copy the algorithm parameters using your existing helper
        deep_copy_params(&target->params, &req_s->params);
    }
}

// Helper to remove a specific ID from a global list
static bool global_del_slice_by_id(ul_dl_slice_conf_t* global_list, uint32_t id_to_remove)
{
    for (uint32_t i = 0; i < global_list->len_slices; i++) {
        if (global_list->slices[i].id == id_to_remove) {
            // Found it! Free memory.
            free(global_list->slices[i].label);
            free(global_list->slices[i].sched);
            if(global_list->slices[i].params.type == SLICE_ALG_SM_V0_EDF && 
               global_list->slices[i].params.u.edf.over) {
                free(global_list->slices[i].params.u.edf.over);
            }

            // Shift remaining elements left
            for (uint32_t j = i; j < global_list->len_slices - 1; j++) {
                global_list->slices[j] = global_list->slices[j+1];
            }

            global_list->len_slices--;
            // Optional: realloc down to save memory, but not strictly required
            return true; // Deleted successfully
        }
    }
    return false; // ID not found
}

// Helper to update Global UE association
static void global_update_ue_assoc(ue_slice_conf_t* global_ues, ue_slice_conf_t* new_assoc)
{
    // Simple logic: If UE RNTI exists, update it. If not, add it.
    // For simplicity here, we will just Append or Update.
    
    if (global_ues->ues == NULL) {
        // Initial alloc
        global_ues->len_ue_slice = new_assoc->len_ue_slice;
        global_ues->ues = calloc(global_ues->len_ue_slice, sizeof(ue_slice_assoc_t));
        memcpy(global_ues->ues, new_assoc->ues, global_ues->len_ue_slice * sizeof(ue_slice_assoc_t));
        return;
    }

    // Append logic (Simulated)
    uint32_t old_len = global_ues->len_ue_slice;
    global_ues->len_ue_slice += new_assoc->len_ue_slice;
    global_ues->ues = realloc(global_ues->ues, global_ues->len_ue_slice * sizeof(ue_slice_assoc_t));
    
    memcpy(&global_ues->ues[old_len], new_assoc->ues, new_assoc->len_ue_slice * sizeof(ue_slice_assoc_t));
}


// =========================================================
// MODIFIED: fill_slice_del
// Now acts as a HANDLER. It reads the 'conf' message 
// and deletes the specified IDs from the Global Data.
// =========================================================
static void fill_slice_del(del_slice_conf_t* conf)
{
  assert(conf != NULL);

  // 1. Ensure Global Data is initialized
  if (!init_flag) {
      slice_ind_data_t dummy; 
      fill_slice_ind_data(&dummy); 
  }

  printf("[E2 Agent] Executing Specific Delete Command...\n");

  // 2. Process DL Deletions requested in 'conf'
  ul_dl_slice_conf_t* global_dl = &slice_data_global.msg.slice_conf.dl;
  for(uint32_t i = 0; i < conf->len_dl; i++) {
      uint32_t id = conf->dl[i];
      printf(" -> Deleting DL Slice ID: %d\n", id);
      global_del_slice_by_id(global_dl, id);
  }

  // 3. Process UL Deletions requested in 'conf'
  ul_dl_slice_conf_t* global_ul = &slice_data_global.msg.slice_conf.ul;
  for(uint32_t i = 0; i < conf->len_ul; i++) {
      uint32_t id = conf->ul[i];
      printf(" -> Deleting UL Slice ID: %d\n", id);
      global_del_slice_by_id(global_ul, id);
  }
}

// =========================================================
// MODIFIED: fill_slice_ctrl
// Now acts as a HANDLER. It looks at the 'ctrl' message type
// and updates the Global Data accordingly.
// =========================================================
void fill_slice_ctrl(slice_ctrl_req_data_t* ctrl)
{
   assert(ctrl != NULL);

   // 1. Ensure Global Data is initialized
   if (!init_flag) {
       slice_ind_data_t dummy; 
       fill_slice_ind_data(&dummy); 
   }

   // 2. Switch on the Specific Command Type provided in the message
   if(ctrl->msg.type == SLICE_CTRL_SM_V0_ADD){
     printf("[E2 Agent] Executing Specific ADD/MOD Command...\n");
     
     // Use the UPSERT helper. 
     // It reads data from 'ctrl' (request) and updates 'slice_data_global'.
     global_upsert_slices(&slice_data_global.msg.slice_conf.dl, &ctrl->msg.u.add_mod_slice.dl);
     global_upsert_slices(&slice_data_global.msg.slice_conf.ul, &ctrl->msg.u.add_mod_slice.ul);

   } 
   else if (ctrl->msg.type == SLICE_CTRL_SM_V0_DEL){
     // Delegate to the modified fill_slice_del function
     fill_slice_del(&ctrl->msg.u.del_slice);

   } 
   else if (ctrl->msg.type == SLICE_CTRL_SM_V0_UE_SLICE_ASSOC){
     printf("[E2 Agent] Executing Specific UE ASSOC Command...\n");
     // Apply the association found in the message
     global_update_ue_assoc(&slice_data_global.msg.ue_slice_conf, &ctrl->msg.u.ue_slice);

   } 
   else {
      printf("[E2 Agent] Unknown Control Command Type: %d\n", ctrl->msg.type);
   }
}