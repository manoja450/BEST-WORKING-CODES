//This code is also working, but in addition, it has chi chi-square/ndf comparison to determine the best fit start time.
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <TGraph.h>
#include <TGaxis.h>

using std::cout;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1100;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 40;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 400; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 0.76;       // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform

// Generate unique output directory with timestamp
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

// SiPM thresholds ( channels 12-21)
const double SIPM_THRESHOLDS[10] = {750, 950, 1200, 1375, 525, 700, 700, 500,450,450}; 
const double FIT_MIN = 1.0; // Fit range min (µs)
const double FIT_MAX = 10.0; // Fit range max (µs)

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    double trigger;        // Trigger type
    double side_sipm_energy; // Side SiPM energy (ADC)
    double top_sipm_energy;  // Top SiPM energy (ADC)
    double all_sipm_energy;  // All SiPM energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
};

// Temporary pulse structure
struct pulse_temp {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// SPE fitting function
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0] - par[1]) / par[2], 2));
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0] - par[4]) / par[5], 2));
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0] - sqrt(2) * par[4]) / sqrt(2 * pow(par[5], 2) - pow(par[2], 2)), 2));
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0] - sqrt(3) * par[4]) / sqrt(3 * pow(par[5], 2) - 2 * pow(par[2], 2)), 2));
    return term1 + term2 + term3 + term4;
}

// Exponential fit function: N0 * exp(-t/tau) + C (t, tau in µs)
Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0] / par[1]) + par[2];
}

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st) != 0) {
        if (mkdir(dirName.c_str(), 0755) != 0) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    TCanvas *c = new TCanvas("c", "SPE Fits", 1200, 800); // Updated to 1200x800
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();

        fitFunc->SetParameters(1000, histMean - histRMS, histRMS / 2,
                              1000, histMean, histRMS,
                              500, 200);

        histArea[i]->Fit(fitFunc, "Q", "", -50, 400);

        mu1[i] = fitFunc->GetParameter(4);
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1 / sqrt(nLEDFlashes[i]), 2));

        // Plot SPE fit
        c->Clear();
        histArea[i]->Draw();
        fitFunc->Draw("same");
        TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
        leg->AddEntry(histArea[i], Form("PMT %d Data", i + 1), "l");
        leg->AddEntry(fitFunc, "SPE Fit", "l");
        leg->AddEntry((TObject*)0, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]), "");
        leg->Draw();
        string plotName = OUTPUT_DIR + Form("/SPE_Fit_PMT%d.png", i + 1);
        c->Update();
        c->SaveAs(plotName.c_str());
        cout << "Saved SPE plot: " << plotName << endl;
        delete leg;
        delete fitFunc;
        delete histArea[i];
    }

    delete c;
    calibFile->Close();
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Check if calibration file exists
    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    // Check if at least one input file exists
    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    // Perform SPE calibration
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    performCalibration(calibFileName, mu1, mu1_err);

    // Print calibration results
    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    // Statistics counters
    int num_muons = 0;
    int num_michels = 0;
    int num_events = 0;

    // Map to track triggerBits counts
    std::map<int, int> trigger_counts;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/100 p.e.", 550, -500, 5000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference ;Time to Previous event(Muon)(#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_side_sipm_muon = new TH1D("side_sipm_muon", "Side SiPM Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_sipm_muon = new TH1D("top_sipm_muon", "Top SiPM Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);

    for (const auto& inputFileName : inputFiles) {
        // Check if input file exists
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            continue;
        }

        // Declaration of leaf types
        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;

        // Set branch addresses
        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        int numEntries = t->GetEntries();
        cout << "Processing " << numEntries << " entries in " << inputFileName << endl;
        double last_muon_time = 0.0;
        std::set<double> michel_muon_times;
        std::vector<std::pair<double, double>> muon_candidates;

        // First pass: Identify Michel electrons and their muon times
        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            // Fill triggerBits histogram and track counts
            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            // Check for out-of-range triggerBits
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0-31) in file " << inputFileName << ", event " << eventID << endl;
            }

            // Initialize pulse
            struct pulse p;
            p.start = nsTime / 1000.0; // Convert ns to µs
            p.end = nsTime / 1000.0;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_sipm_energy = 0;
            p.top_sipm_energy = 0;
            p.all_sipm_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_sipm_energy, top_sipm_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> sipm_energies(10, 0); // Channels 12-21

            for (int iChan = 0; iChan < 23; iChan++) {
                // Fill waveform histogram
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                // Check beam status (channel 22)
                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                // Pulse detection
                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0; // Convert ns to µs
                            pt.peak = iChan <= 11 && mu1[iChan] > 0 ? peak / mu1[iChan] : peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) {
                                pt.energy = mu1[iChan] > 0 ? pulseEnergy / mu1[iChan] : 0;
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            peakBin = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                // Store energy for SiPM panels (ADC)
                if (iChan >= 12 && iChan <= 19) {
                    side_sipm_energy.push_back(allPulseEnergy);
                    sipm_energies[iChan - 12] = allPulseEnergy;
                } else if (iChan >= 20 && iChan <= 21) {
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    top_sipm_energy.push_back(allPulseEnergy * factor);
                    sipm_energies[iChan - 12] = allPulseEnergy * factor;
                }

                // Check for pulses at waveform end
                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            // Aggregate pulse properties
            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_sipm_energy = std::accumulate(side_sipm_energy.begin(), side_sipm_energy.end(), 0.0);
            p.top_sipm_energy = std::accumulate(top_sipm_energy.begin(), top_sipm_energy.end(), 0.0);
            p.all_sipm_energy = p.side_sipm_energy + p.top_sipm_energy;

            // Check timing consistency
            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);

            // Muon detection
            bool sipm_hit = false;
            for (int i = 0; i < 10; i++) {
                if (sipm_energies[i] > SIPM_THRESHOLDS[i]) {
                    sipm_hit = true;
                    break;
                }
            }

            if ((p.energy > MUON_ENERGY_THRESHOLD && sipm_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && sipm_hit)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
                h_side_sipm_muon->Fill(p.side_sipm_energy);
                h_top_sipm_muon->Fill(p.top_sipm_energy);
            }

            // Michel electron detection
            double dt = p.start - last_muon_time;
            bool sipm_low = true;
            for (int i = 0; i < 10; i++) {
                if (sipm_energies[i] > SIPM_THRESHOLDS[i]) {
                    sipm_low = false;
                    break;
                }
            }

            // Define common Michel electron criteria
            bool is_michel_candidate = p.energy >= MICHEL_ENERGY_MIN &&
                                      p.energy <= MICHEL_ENERGY_MAX &&
                                      dt >= MICHEL_DT_MIN &&
                                      dt <= MICHEL_DT_MAX &&
                                      p.number >= 8 &&
                                      sipm_low &&
                                      p.trigger != 1 &&
                                      p.trigger != 4 &&
                                      p.trigger != 8 &&
                                      p.trigger != 16;
                        
            // Apply additional cut for dt and energy_vs_dt plots
            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                // Fill Michel energy histogram with original criteria
                h_energy_vs_dt->Fill(dt, p.energy);
                h_michel_energy->Fill(p.energy);
            }

            if (is_michel_for_dt) {
                // Fill dt and energy_vs_dt histograms with stricter energy cut
                h_dt_michel->Fill(dt);
            }

            p.last_muon_time = last_muon_time;
        }

        // Second pass: Fill h_muon_energy for muons associated with Michel electrons
        for (const auto& muon : muon_candidates) {
            if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
                h_muon_energy->Fill(muon.second);
            }
        }

        // Print stats to console
        cout << "File " << inputFileName << " Statistics:\n";
        cout << "Total Events: " << num_events << "\n";
        cout << "Muons Detected: " << num_muons << "\n";
        cout << "Michel Electrons Detected: " << num_michels << "\n";
        cout << "------------------------\n";

        f->Close();

        num_events = 0;
        num_muons = 0;
        num_michels = 0;
    }

    // Print triggerBits distribution
    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";

    // Generate analysis plots
    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // Muon Energy
    c->Clear();
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel Energy
    c->Clear();
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Michel dt with exponential fit - FIXED
    c->Clear();
    h_dt_michel->SetLineWidth(2);
    h_dt_michel->SetLineColor(kBlack);
    h_dt_michel->GetXaxis()->SetTitle("Time to previous event (Muon) [#mus]");
    h_dt_michel->Draw("HIST");  // Draw histogram first

    TF1* expFit = nullptr;
    if (h_dt_michel->GetEntries() > 5) {
        // Initial parameter estimates
        double integral = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
        double bin_width = h_dt_michel->GetBinWidth(1);
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN);
        double C_init = 0;
        
        // Estimate constant background from last bins (12-16 µs)
        int bin_12 = h_dt_michel->FindBin(12.0);
        int bin_16 = h_dt_michel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_12; i <= bin_16; i++) {
            double content = h_dt_michel->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        // Create and configure fit function
        expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, 2.2, C_init);
        expFit->SetParLimits(0, 0, N0_init * 100);
        expFit->SetParLimits(1, 0.1, 20.0);
        expFit->SetParLimits(2, -C_init * 10, C_init * 10);
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetLineColor(kRed);
        expFit->SetLineWidth(3);

        // Perform fit and draw on top
        int fitStatus = h_dt_michel->Fit(expFit, "RE+", "SAME", FIT_MIN, FIT_MAX);
        expFit->Draw("SAME");  // Explicitly draw the fit function
        
        // Update stats box
        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_michel->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.6);
            stats->SetY2NDC(0.9);
            stats->SetTextColor(kRed);
            stats->Clear();
            stats->AddText("DeltaT");
            stats->AddText(Form("#tau = %.4f #pm %.4f #mus", expFit->GetParameter(1), expFit->GetParError(1)));
            stats->AddText(Form("#chi^{2}/NDF = %.4f", expFit->GetChisquare() / expFit->GetNDF()));
            stats->AddText(Form("N_{0} = %.1f #pm %.1f", expFit->GetParameter(0), expFit->GetParError(0)));
            stats->AddText(Form("C = %.1f #pm %.1f", expFit->GetParameter(2), expFit->GetParError(2)));
            stats->Draw();
        }
        
        // Print fit results
        double N0 = expFit->GetParameter(0);
        double N0_err = expFit->GetParError(0);
        double tau = expFit->GetParameter(1);
        double tau_err = expFit->GetParError(1);
        double C = expFit->GetParameter(2);
        double C_err = expFit->GetParError(2);
        double chi2 = expFit->GetChisquare();
        int ndf = expFit->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Michel dt, " << FIT_MIN << "-" << FIT_MAX << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("N₀ = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_michel has insufficient entries (" << h_dt_michel->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c->Update();
    c->Modified();
    c->RedrawAxis();
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;
    
    // Clean up fit function
    if (expFit) {
        delete expFit;
        expFit = nullptr;
    }

    // ======================================================================
    // [Revised Section] Compare different exponential fit start times
    // ======================================================================
    if (h_dt_michel->GetEntries() > 5) {
        // Clear any existing functions from previous fits
        h_dt_michel->GetListOfFunctions()->Clear();
        
        // Define fit start times to try (1.0 to 4.0 μs in 0.5 μs steps)
        std::vector<double> fit_starts;
        for (double s = 1.0; s <= 4.0; s += 0.5) {
            fit_starts.push_back(s);
        }
        
        // Vectors to store fit results
        std::vector<double> taus, tau_errs, chi2ndfs;
        int best_index = -1;
        double min_chi2ndf = 1e9;
        
        // Perform fits for each start time
        for (int i = 0; i < fit_starts.size(); i++) {
            double fit_start = fit_starts[i];
            double fit_end = 16.0;
            
            TF1* expFit_var = new TF1(Form("expFit_var_%.1f", fit_start), ExpFit, fit_start, fit_end, 3);
            
            // Estimate background from last bins (12-16 μs)
            double C_init = 0;
            int bin_12 = h_dt_michel->FindBin(12.0);
            int bin_16 = h_dt_michel->FindBin(16.0);
            double min_content = 1e9;
            for (int bin = bin_12; bin <= bin_16; bin++) {
                double content = h_dt_michel->GetBinContent(bin);
                if (content > 0 && content < min_content) min_content = content;
            }
            if (min_content < 1e9) C_init = min_content;
            else C_init = 0.1;
            
            // Estimate initial parameters
            double integral = h_dt_michel->Integral(h_dt_michel->FindBin(fit_start), h_dt_michel->FindBin(fit_end));
            double bin_width = h_dt_michel->GetBinWidth(1);
            double N0_init = (integral * bin_width - C_init * (fit_end - fit_start)) / 2.2;
            if (N0_init < 0) N0_init = 100;
            
            // Configure fit function
            expFit_var->SetParameters(N0_init, 2.2, C_init);
            expFit_var->SetParNames("N_{0}", "#tau", "C");
            
            // Perform fit (quietly)
            h_dt_michel->Fit(expFit_var, "QRN+", "", fit_start, fit_end);
            
            // Record results
            double tau = expFit_var->GetParameter(1);
            double tau_err = expFit_var->GetParError(1);
            double chi2 = expFit_var->GetChisquare();
            int ndf = expFit_var->GetNDF();
            double chi2ndf = (ndf > 0) ? chi2 / ndf : 999;
            
            taus.push_back(tau);
            tau_errs.push_back(tau_err);
            chi2ndfs.push_back(chi2ndf);
            
            if (chi2ndf < min_chi2ndf) {
                min_chi2ndf = chi2ndf;
                best_index = i;
            }
            
            delete expFit_var;
        }
        
        // Create comparison plot
        TCanvas* c_comp = new TCanvas("c_comp", "Fit Start Time Comparison", 1200, 800); // Updated to 1200x800
        c_comp->SetGrid();
        
        // Create pad for the main plot
        TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1);
        pad->Draw();
        pad->cd();
        
        // Create graphs
        TGraph* g_chi2 = new TGraph(fit_starts.size(), &fit_starts[0], &chi2ndfs[0]);
        TGraph* g_tau = new TGraph(fit_starts.size(), &fit_starts[0], &taus[0]);
        
        // Configure chi2 graph (left axis)
        g_chi2->SetTitle("Fit Start Time Comparison");
        g_chi2->GetXaxis()->SetTitle("Fit Start Time (#mus)");
        g_chi2->GetYaxis()->SetTitle("#chi^{2}/ndf");
        g_chi2->SetMarkerStyle(20);
        g_chi2->SetMarkerColor(kBlue);
        g_chi2->SetLineColor(kBlue);
        g_chi2->SetLineWidth(2);
        
        // Configure tau graph (right axis)
        g_tau->SetMarkerStyle(22);
        g_tau->SetMarkerColor(kRed);
        g_tau->SetLineColor(kRed);
        g_tau->SetLineWidth(2);
        
        // Draw chi2 first to establish the frame
        g_chi2->Draw("APL");
        
        // Create right axis
        pad->Update();
        double ymin = pad->GetUymin();
        double ymax = pad->GetUymax();
        
        // Scale tau values to match chi2 plot range
        double tau_min = *min_element(taus.begin(), taus.end());
        double tau_max = *max_element(taus.begin(), taus.end());
        double scale = (ymax - ymin)/(tau_max - tau_min);
        double offset = ymin - tau_min * scale;
        
        // Scale the tau graph
        for (int i = 0; i < g_tau->GetN(); i++) {
            double x, y;
            g_tau->GetPoint(i, x, y);
            g_tau->SetPoint(i, x, y * scale + offset);
        }
        
        // Draw tau graph on same pad
        g_tau->Draw("PL same");
        
        // Create right axis
        TGaxis* axis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                                 gPad->GetUxmax(), gPad->GetUymax(),
                                 tau_min, tau_max, 510, "+L");
        axis->SetLineColor(kRed);
        axis->SetLabelColor(kRed);
        axis->SetTitle("#tau (#mus)");
        axis->SetTitleColor(kRed);
        axis->Draw();
        
        // Add legend
        TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
        leg->AddEntry(g_chi2, "#chi^{2}/ndf", "lp");
        leg->AddEntry(g_tau, "#tau", "lp");
        leg->Draw();
        
        // Save plot
        string compPlotName = OUTPUT_DIR + "/FitStartComparison.png";
        c_comp->SaveAs(compPlotName.c_str());
        cout << "Saved comparison plot: " << compPlotName << endl;
        
        // Clean up
        delete g_chi2;
        delete g_tau;
        delete leg;
        delete axis;
        delete pad;
        delete c_comp;
    }
    else {
        cout << "Skipping fit start comparison - insufficient entries in dt histogram" << endl;
    }
    // ======================================================================

    // Energy vs dt
    c->Clear();
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Side SiPM Muon
    c->Clear();
    h_side_sipm_muon->SetLineColor(kMagenta);
    h_side_sipm_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Side_SiPM_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Top SiPM Muon
    c->Clear();
    h_top_sipm_muon->SetLineColor(kCyan);
    h_top_sipm_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Top_SiPM_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Trigger Bits Distribution
    c->Clear();
    h_trigger_bits->SetLineColor(kGreen);
    h_trigger_bits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Clean up
    delete h_muon_energy;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_sipm_muon;
    delete h_top_sipm_muon;
    delete h_trigger_bits;
    delete c;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png)" << endl;
    return 0;
}
