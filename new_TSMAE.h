#ifndef TSMAE_H
#define TSMAE_H

#include "systemc.h"
#include <iostream>
#include "LSTM_neural.h"
#include "MemoryModule.h"
#include "Mem_de_driver.h"
#include "Output_layer.h"
#include "decoder.h"
#include "weight.h"

SC_MODULE(TSMAE) {
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<float> x; 
    sc_in<bool> rdy_to_receiv;
    sc_in<bool> rdy_to_send;
	sc_in<bool> rdy_to_output;
  	sc_out<bool> is_rdy_to_output;
    sc_out<bool> is_rdy_to_send;
    sc_out<bool> is_rdy_to_receiv;
    
    sc_out<float> x_recon;

    // Modules
    LSTM_neural *encoder;
    Decoder *decoder;
    MemoryModule *memory_module;
    Mem_de_driver *driver;
    Output_layer *output_layer;
   

    // Internal registers
    float x_recon_reg[32];
    float loss;
    int count;

    // Signals
    sc_signal<bool> output_done_ques, output_done_ans;
    sc_signal<bool> en_mem_ques, en_mem_ans;
    sc_signal<bool> mem_driver_ques, mem_driver_ans;
    sc_signal<bool> driver_de_ques, driver_de_ans;
    sc_signal<bool> de_out_ques, de_out_ans;
    sc_signal<bool> queue_read_en;
    sc_signal<bool> output_en, for_free, en_for_free, de_for_free;
    sc_signal<float> x_recon_wire;
    sc_signal<float> queue_out;
    sc_signal<float> en_mem_hidden[10];
    sc_signal<float> de_out_hidden[10];
    sc_signal<float> mem_driver_latent[10];
    sc_signal<float> driver_de_series[10];

    SC_CTOR(TSMAE) {
        output_en.write(1);

        // Module init
        encoder = new LSTM_neural("encoder");
        decoder = new Decoder("decoder");
        memory_module = new MemoryModule("memory_module");
        output_layer = new Output_layer("output_layer");
        driver = new Mem_de_driver("driver");

        // Encoder
        encoder->clk(clk); encoder->rst(rst);
        encoder->rdy_to_receiv(rdy_to_receiv);
        encoder->is_rdy_to_receiv(is_rdy_to_receiv);
        encoder->rdy_to_output(rdy_to_output);
        encoder->is_rdy_to_output(is_rdy_to_output);
        encoder->x(x);
        encoder->rdy_to_send(en_mem_ans);
        encoder->is_rdy_to_send(en_mem_ques);
        for(int i = 0; i < 10; i++) encoder->ht[i](en_mem_hidden[i]);

        // Memory
        memory_module->clk(clk); memory_module->rst(rst);
        memory_module->rdy_to_receiv(en_mem_ques);
        memory_module->is_rdy_to_receiv(en_mem_ans);
        for (int i = 0; i < 10 ; i++) {
            memory_module->last_ht[i](en_mem_hidden[i]);
            memory_module->latent[i](mem_driver_latent[i]);
        }
        memory_module->rdy_to_send(mem_driver_ans);
        memory_module->is_rdy_to_send(mem_driver_ques);

        // Driver
        driver->clk(clk); driver->rst(rst);
        for (int i = 0; i < 10; i++) {
            driver->latent[i](mem_driver_latent[i]);
            driver->latent_series[i](driver_de_series[i]);
        }
        driver->is_rdy_to_receiv(mem_driver_ans);
        driver->rdy_to_receiv(mem_driver_ques);
        driver->is_rdy_to_send(driver_de_ques);
        driver->rdy_to_send(driver_de_ans);

        // Decoder
        decoder->clk(clk); decoder->rst(rst);
        decoder->rdy_to_receiv(driver_de_ques);
        decoder->is_rdy_to_receiv(driver_de_ans);
        decoder->rdy_to_output(de_out_ans);
        decoder->is_rdy_to_output(de_out_ques);
        decoder->rdy_to_send(de_for_free);
        decoder->is_rdy_to_send(for_free);
        for(int i = 0; i < 10; i++) {
            decoder->ht[i](de_out_hidden[i]);
            decoder->x[i](driver_de_series[i]);
        }

        // Output layer
        output_layer->clk(clk); output_layer->rst(rst);
        output_layer->rdy_to_receiv(de_out_ques);
        output_layer->is_rdy_to_receiv(de_out_ans);
        output_layer->rdy_to_send(rdy_to_send);
        output_layer->is_rdy_to_send(is_rdy_to_send);
        for(int i = 0; i < 10 ; i++) output_layer->ht[i](de_out_hidden[i]);
        output_layer->x_recon(x_recon);
       
    }

    void load_weights();
    float pow(float x);
};

// ============================================
// ========== IMPLEMENTATION BELOW ============
// ============================================


void TSMAE::load_weights() {
    output_layer->output_bias = output_bias;
    for (int i = 0; i < 10; i++) {
        decoder->c_bias_reg[i] = decoder_bias_hh_cell[i];
        decoder->f_bias_reg[i] = decoder_bias_hh_forget[i];
        decoder->i_bias_reg[i] = decoder_bias_hh_input[i];
        decoder->o_bias_reg[i] = decoder_bias_hh_output[i];
        decoder->xc_bias_reg[i] = decoder_bias_ih_cell[i];
        decoder->xf_bias_reg[i] = decoder_bias_ih_forget[i];
        decoder->xi_bias_reg[i] = decoder_bias_ih_input[i];
        decoder->xo_bias_reg[i] = decoder_bias_ih_output[i];

        encoder->c_bias_reg[i] = encoder_bias_hh_cell[i];
        encoder->f_bias_reg[i] = encoder_bias_hh_forget[i];
        encoder->i_bias_reg[i] = encoder_bias_hh_input[i];
        encoder->o_bias_reg[i] = encoder_bias_hh_output[i];
        encoder->xc_bias_reg[i] = encoder_bias_ih_cell[i];
        encoder->xf_bias_reg[i] = encoder_bias_ih_forget[i];
        encoder->xi_bias_reg[i] = encoder_bias_ih_input[i];
        encoder->xo_bias_reg[i] = encoder_bias_ih_output[i];

        encoder->xc_weight_matrix[i] = encoder_weight_ih_cell[i];
        encoder->xf_weight_matrix[i] = encoder_weight_ih_forget[i];
        encoder->xi_weight_matrix[i] = encoder_weight_ih_input[i];
        encoder->xo_weight_matrix[i] = encoder_weight_ih_output[i];

        output_layer->output_weight[i] = output_layer_weight_row_0[i];

        for (int j = 0; j < 10; j++) {
            decoder->c_weight_matrix[i][j] = decoder_weight_hh_cell[i][j];
            decoder->f_weight_matrix[i][j] = decoder_weight_hh_forget[i][j];
            decoder->i_weight_matrix[i][j] = decoder_weight_hh_input[i][j];
            decoder->o_weight_matrix[i][j] = decoder_weight_hh_output[i][j];
            decoder->xc_weight_matrix[i][j] = decoder_weight_ih_cell[i][j];
            decoder->xf_weight_matrix[i][j] = decoder_weight_ih_forget[i][j];
            decoder->xi_weight_matrix[i][j] = decoder_weight_ih_input[i][j];
            decoder->xo_weight_matrix[i][j] = decoder_weight_ih_output[i][j];

            encoder->c_weight_matrix[i][j] = encoder_weight_hh_cell[i][j];
            encoder->f_weight_matrix[i][j] = encoder_weight_hh_forget[i][j];
            encoder->i_weight_matrix[i][j] = encoder_weight_hh_input[i][j];
            encoder->o_weight_matrix[i][j] = encoder_weight_hh_output[i][j];
        }
    }

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 10; j++) {
            memory_module->mem_item[i][j] = memory_matrix[i][j];
        }
    }
}

float TSMAE::pow(float x) {
    return x * x;
}

#endif // TSMAE_H
