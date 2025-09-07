#include <systemc.h>
#include <iostream>

SC_MODULE(Decoder) {
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> rdy_to_receiv;
    sc_in<bool> rdy_to_send;
    sc_in<bool> rdy_to_output;

    sc_in<float> x[10];
    sc_out<float> ht[10];

    sc_out<bool> is_rdy_to_send;
    sc_out<bool> is_rdy_to_output;
    sc_out<bool> is_rdy_to_receiv;

    sc_signal<float> c_feedback[10];
    sc_signal<float> h_feedback[10];

    float f_bias_reg[10], i_bias_reg[10], o_bias_reg[10], c_bias_reg[10];
    float xf_bias_reg[10], xi_bias_reg[10], xo_bias_reg[10], xc_bias_reg[10];

    float xf_weight_matrix[10][10], xi_weight_matrix[10][10];
    float xo_weight_matrix[10][10], xc_weight_matrix[10][10];

    float xf_mul[10], xi_mul[10], xo_mul[10], xc_mul[10];
    float f_weight_matrix[10][10], i_weight_matrix[10][10];
    float o_weight_matrix[10][10], c_weight_matrix[10][10];
    float f_mul[10], i_mul[10], o_mul[10], c_mul[10];

    float ht_1_reg[10], ct_1_reg[10];
    float ht_reg[10], ct_reg[10];
    int hidden_count = 0;

    SC_CTOR(Decoder) {
        SC_CTHREAD(lstm_process, clk.pos());
        async_reset_signal_is(rst, true);
    }

    void lstm_process();
    float exp(float x);
    float sigmoid(float x);
    float tanh(float x);
};

// ======= Hàm được định nghĩa ngoài SC_MODULE =======

void Decoder::lstm_process() {
    while (true) {
        if (rst.read()) {
            hidden_count = 0;
            

            for (int i = 0; i < 10; i++) {
                xf_mul[i] = xi_mul[i] = xo_mul[i] = xc_mul[i] = 0;
                f_mul[i] = i_mul[i] = o_mul[i] = c_mul[i] = 0;
                ht_1_reg[i] = ct_1_reg[i] = ht_reg[i] = ct_reg[i] = 0;
                ht[i].write(0);
                h_feedback[i].write(0);
                c_feedback[i].write(0);
                wait(); // mỗi reset = 1 chu kỳ
            }
        } else {
            is_rdy_to_receiv.write(1);
            do { wait(); } while (!rdy_to_receiv.read());
            is_rdy_to_receiv.write(0);

            if (hidden_count == 0) {
                for (int i = 0; i < 10; i++) {
                    ht_1_reg[i] = 0;
                    ct_1_reg[i] = 0;
                    wait();
                }
            } else {
                for (int i = 0; i < 10; i++) {
                    ht_1_reg[i] = h_feedback[i].read();
                    ct_1_reg[i] = c_feedback[i].read();
                    wait();
                }
            }

            for (int i = 0; i < 10; i++) {
                xf_mul[i] = xi_mul[i] = xo_mul[i] = xc_mul[i] = 0;
                wait();
                for (int j = 0; j < 10; j++) {
                    float xj = x[j].read();
                    xf_mul[i] += xj * xf_weight_matrix[i][j];
                    xi_mul[i] += xj * xi_weight_matrix[i][j];
                    xo_mul[i] += xj * xo_weight_matrix[i][j];
                    xc_mul[i] += xj * xc_weight_matrix[i][j];
                    wait(); // mỗi phép nhân + cộng = 1 chu kỳ
                }
            }

            for (int i = 0; i < 10; i++) {
                f_mul[i] = i_mul[i] = o_mul[i] = c_mul[i] = 0;
                wait();
                for (int j = 0; j < 10; j++) {
                    f_mul[i] += ht_1_reg[j] * f_weight_matrix[j][i];
                    i_mul[i] += ht_1_reg[j] * i_weight_matrix[j][i];
                    o_mul[i] += ht_1_reg[j] * o_weight_matrix[j][i];
                    c_mul[i] += ht_1_reg[j] * c_weight_matrix[j][i];
                    wait(); // mỗi phép nhân + cộng = 1 chu kỳ
                }
            }

            for (int i = 0; i < 10; i++) {
                f_mul[i] += xf_mul[i] + f_bias_reg[i] + xf_bias_reg[i];
                i_mul[i] += xi_mul[i] + i_bias_reg[i] + xi_bias_reg[i];
                o_mul[i] += xo_mul[i] + o_bias_reg[i] + xo_bias_reg[i];
                c_mul[i] += xc_mul[i] + c_bias_reg[i] + xc_bias_reg[i];
                wait();
            }

            for (int i = 0; i < 10; i++) {
                f_mul[i] = sigmoid(f_mul[i]);
                i_mul[i] = sigmoid(i_mul[i]);
                o_mul[i] = sigmoid(o_mul[i]);
                c_mul[i] = tanh(c_mul[i]);
                wait();
            }

            for (int i = 0; i < 10; i++) {
                ct_reg[i] = f_mul[i] * ct_1_reg[i] + i_mul[i] * c_mul[i];
                ht_reg[i] = o_mul[i] * tanh(ct_reg[i]);
                wait();
            }

            for (int i = 0; i < 10; i++) {
                ht[i].write(ht_reg[i]);
                h_feedback[i].write(ht_reg[i]);
                c_feedback[i].write(ct_reg[i]);
                wait(); // mỗi write = 1 chu kỳ
            }

            is_rdy_to_output.write(1);
            do { wait(); } while (!rdy_to_output.read());
            is_rdy_to_output.write(0);

            hidden_count++;
            if (hidden_count == 140) {
                hidden_count = 0;
                is_rdy_to_send.write(1);
                do { wait(); } while (!rdy_to_send.read());
                is_rdy_to_send.write(0);
            }
        }
        wait(); // mỗi vòng lặp luôn chờ nhịp
    }
}

float Decoder::exp(float x) {
    float result = 1.0, term = 1.0;
    for (int i = 1; i <= 10; ++i) {
        term *= x / i;
        result += term;
    }
    return result;
}

float Decoder::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float Decoder::tanh(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
