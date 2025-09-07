#include <systemc.h>
#include <iostream>

SC_MODULE(Output_layer) {
    sc_in<bool> rst;
    sc_in<bool> clk;
    sc_in<float> ht[10];
    sc_out<float> x_recon;
    sc_in<bool> rdy_to_receiv;
    sc_in<bool> rdy_to_send;
    sc_out<bool> is_rdy_to_send;
    sc_out<bool> is_rdy_to_receiv;

    float output_weight[10];
    float output_bias;
    float mul;
    float ht_reg[10];

    SC_CTOR(Output_layer) {
        SC_CTHREAD(process, clk.pos());
        async_reset_signal_is(rst, true);
    }

    void process();
};

// ========== Implementation ==========

void Output_layer::process() {
    while (true) {
        if (rst.read()) {
            
            wait(); // thiết lập tín hiệu = 1 chu kỳ

            for (int i = 0; i < 10; i++) {
                ht_reg[i] = 0;
                wait(); // reset từng phần tử = 1 chu kỳ
            }

            mul = 0;
            wait(); // reset tích đầu ra
        } else {
            is_rdy_to_receiv.write(1);
            wait(); // báo sẵn sàng nhận

            do {
                wait(); // chờ module trước cho phép
            } while (!rdy_to_receiv.read());

            is_rdy_to_receiv.write(0);
            wait(); // clear lại tín hiệu

            for (int i = 0; i < 10; i++) {
                ht_reg[i] = ht[i].read();
                wait(); // đọc từng phần tử ht = 1 chu kỳ
            }

            mul = 0;
            wait(); // reset tích đầu ra

            for (int i = 0; i < 10; i++) {
                mul += ht_reg[i] * output_weight[i];
                wait(); // mỗi nhân + cộng = 1 chu kỳ
            }

            mul += output_bias;
            wait(); // cộng bias = 1 chu kỳ

            x_recon.write(mul);
            wait(); // ghi output = 1 chu kỳ

            is_rdy_to_send.write(1);
            wait(); // báo ready = 1 chu kỳ

            do {
                wait(); // chờ module sau sẵn sàng nhận
            } while (!rdy_to_send.read());

            is_rdy_to_send.write(0);
            wait(); // clear lại tín hiệu
        }

        wait(); // cuối vòng lặp
    }
}
