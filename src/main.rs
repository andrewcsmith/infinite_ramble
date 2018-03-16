extern crate sample;
extern crate hound;
extern crate vox_box;
extern crate rulinalg;
extern crate random_choice;

use std::error::Error;
use std::f64;

use sample::{Sample, ToSampleSlice, FromSampleSlice, window};
use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rulinalg::norm::Euclidean;
use random_choice::random_choice;
use vox_box::spectrum::MFCC;

const NCOEFFS: usize = 40;
const BIN: usize = 2048;
const HOP: usize = 2048;
const DUCK_VAL: f64 =  0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Duck;

impl window::Type for Duck {
    fn at_phase<S: Sample>(phase: S) -> S {
        match phase.to_float_sample() {
            x if x > DUCK_VAL.to_sample() => {
                let diff: S::Float = 1.0.to_sample::<S::Float>() - x;
                (diff * (1.0 / (1.0 - DUCK_VAL)).log10().to_sample()).to_sample::<S>()
            }
            x if x < (-DUCK_VAL).to_sample() => {
                let diff: S::Float = 1.0.to_sample::<S::Float>() + x;
                (diff * (1.0 / (1.0 - DUCK_VAL)).log10().to_sample()).to_sample::<S>()
            }
            _ => {
                S::identity().to_sample::<S>()
            }
        }
    }
}

fn main() {
    go().unwrap();
}

fn go() -> Result<(), Box<Error>> {
    // mono file so I don't care whether it's interleaved
    let mut reader = hound::WavReader::open("full_source.wav")?;
    let samples: Vec<f64> = reader.samples::<i16>()
        .map(|s| s.unwrap().to_sample())
        .collect();

    let mfccs = analyze_mfccs(44_100.0, &samples[..]);
    let sim = self_similarity_matrix(mfccs.clone(), mfccs);
    // println!("sim: {:?}", sim);

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();

    // Looking for 512 frames, not 512 samples
    let frames: Vec<usize> = (0..sim.rows()).collect();
    let mut frame_index = 0;
    let mut chooser = random_choice();

    let max = 500;
    let c = 2.0;

    let frame_indices: Vec<usize> = (0..max).map(|coeff| {
        // let weights: Vec<f64> = sim.row(coeff).iter()
        //     .map(|s| {
        //         // let angular = 1.0 - (s.acos() * f64::consts::FRAC_1_PI);
        //         // let e = ((max - coeff) as f64 / max as f64) * c - (c * 0.5);
        //         // (s + 1.0).powf(e)
        //         *s
        //     }).collect();
        // // frame_index = *chooser.random_choice_f64(&frames, &weights, 1)[0];
        sim.row(coeff).iter().enumerate()
            .fold((0, -1.0), |(mut midx, mut m), (sidx, s)| {
                if *s > m {
                    m = *s;
                    midx = sidx;
                }
                (midx, m)
            }).0
    }).collect();

    let win: window::Window<[f64; 1], window::Hanning> = window::Window::new(BIN);
    for f in frame_indices.chunks(2) {
        let chosen_left = samples[f[0]*HOP..f[0]*HOP+BIN]
            .iter().zip(win.clone()).map(|(s, w)| s * w[0]);
        let chosen_right = samples[f[1]*HOP..f[1]*HOP+BIN]
            .iter().zip(win.clone()).map(|(s, w)| s * w[0]);

        for (l, r) in chosen_left.zip(chosen_right) {
            writer.write_sample(l.to_sample::<i16>())?;
            writer.write_sample(r.to_sample::<i16>())?;
        }
    }

    writer.finalize()?;

    Ok(())
}

fn self_similarity_matrix(input: Matrix<f64>, other: Matrix<f64>) -> Matrix<f64> {
    assert_eq!(input.cols(), other.cols());
    let norms1 = input.row_iter().map(|r| r.norm(Euclidean));
    let norms2 = other.row_iter().map(|r| r.norm(Euclidean));

    let t = other.transpose();
    let mut d = input.clone() * t;

    for ((mut row, norm1), norm2) in d.row_iter_mut().zip(norms1).zip(norms2) {
        let n = norm1 * norm2;
        for x in row.raw_slice_mut().iter_mut() {
            *x /= n;
        }
    }

    d
}

fn analyze_mfccs(sample_rate: f64, samples: &[f64]) -> Matrix<f64> {
    let mfcc_calc = |frame: &[f64]| -> [f64; NCOEFFS] { 
        let mut mfccs = [0f64; NCOEFFS];
        let m = frame.mfcc(NCOEFFS, (100., 8000.), sample_rate as f64);
        for (i, c) in m.iter().enumerate() {
            mfccs[i] = *c;
        }
        mfccs
    };

    let mut frame_buffer: Vec<f64> = Vec::with_capacity(BIN);

    // A single vector of mfccs
    let v: Vec<f64> = window::Windower::hanning(
        <&[[f64; 1]]>::from_sample_slice(samples).unwrap(), BIN, HOP)
        .map(|frame| {
            for s in frame.take(BIN) {
                frame_buffer.extend_from_slice(<&[f64]>::to_sample_slice(&s[..]));
            }
            let mfccs = mfcc_calc(&frame_buffer[..]);
            frame_buffer.clear();
            mfccs
        })
        .fold(Vec::<f64>::with_capacity(samples.len() * NCOEFFS / BIN), |mut acc, v| {
            acc.extend_from_slice(&v[..]);
            acc
        });

    Matrix::new(v.len() / NCOEFFS, NCOEFFS, v)
}
