extern crate sample;
extern crate hound;
extern crate vox_box;
extern crate ndarray;
extern crate ndarray_parallel;
extern crate random_choice;
extern crate serde;
extern crate serde_pickle;

use std::error::Error;
use std::f64;
use std::fs::File;
use std::path::Path;
use std::io::prelude::*;

use sample::{Sample, ToSampleSlice, FromSampleSlice, window};
use random_choice::random_choice;
use ndarray::{ArrayView, Array, Array2, AsArray, Axis, Ix2};
use ndarray_parallel::NdarrayIntoParallelIterator;
use ndarray_parallel::prelude::*;
use vox_box::spectrum::MFCC;

const NCOEFFS: usize = 32;
const BIN: usize = 2048;
const HOP: usize = 512;
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
    let mut reader = hound::WavReader::open("source_minus_end.wav")?;
    let me_samples: Vec<f64> = reader.samples::<i16>()
        .map(|s| s.unwrap().to_sample())
        .collect();
    let mut reader = hound::WavReader::open("target.wav")?;
    let you_samples: Vec<f64> = reader.samples::<i16>()
        .map(|s| s.unwrap().to_sample())
        .collect();

    let me_mfccs = analyze_mfccs(44_100.0, &me_samples[..]);
    let you_mfccs = analyze_mfccs(44_100.0, &you_samples[..]);
    let rows = me_mfccs.len() / NCOEFFS;
    { 
        let mfccs = ArrayView::from_shape([rows, NCOEFFS], &me_mfccs)?;
        let serialized = serde_pickle::to_vec(&mfccs, true)?;
        let mut pickled = File::create(Path::new("mfccs.pk"))?;
        pickled.write_all(&serialized)?;
    }

    let sim = similarity_matrix(&me_mfccs, &you_mfccs)?;
    { 
        let serialized = serde_pickle::to_vec(&sim, true)?;
        let mut pickled = File::create(Path::new("sim.pk"))?;
        pickled.write_all(&serialized)?;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("out.wav", spec).unwrap();

    let max = sim.cols();
    let c = 0.0;

    // The list of source frames
    let frames: Vec<usize> = (0..sim.rows()).collect();

    let mut chooser = random_choice();
    let mut frame_index = 0;

    let frame_indices: Vec<usize> = (0..max).map(|coeff| {
        let weights: Vec<f64> = sim.t().row(coeff).iter()
            .map(|s| {
                let angular = s.acos() * f64::consts::FRAC_1_PI;
                // let e = ((max - coeff) as f64 / max as f64) * c - (c * 0.5);
                (1.0 - angular).powi(1000)
            }).collect();
        frame_index = *chooser.random_choice_f64(&frames, &weights, 1)[0];
        frame_index
    }).collect();

    let win: window::Window<[f64; 1], window::Hanning> = window::Window::new(BIN);

    let out_len = frame_indices.len() * HOP + (BIN - HOP);
    let mut out = vec![0f64; out_len];

    for (idx, f) in frame_indices.iter().enumerate() {
        let samps = &me_samples[f*HOP..f*HOP+BIN];
        let out_start = idx * HOP;
        for (out_idx, (s, w)) in samps.iter().zip(win.clone()).enumerate() {
            out[out_start+out_idx] += s * w[0];
        }
    }

    for s in out {
        writer.write_sample(s.to_sample::<i16>())?;
    }
    writer.finalize()?;

    Ok(())
}

fn similarity_matrix(me_slice: &[f64], you_slice: &[f64]) -> Result<Array2<f64>, Box<Error>>
{
    let me_shape = [me_slice.len() / NCOEFFS, NCOEFFS];
    let you_shape = [you_slice.len() / NCOEFFS, NCOEFFS];
    let me = ArrayView::from_shape(me_shape, me_slice)?;
    let you = ArrayView::from_shape(you_shape, you_slice)?;

    let mut me_norms = Vec::new();
    me.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.fold(0.0, |memo, val| val.powi(2) + memo).sqrt())
        .collect_into_vec(&mut me_norms);

    let mut you_norms = Vec::new();
    you.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.fold(0.0, |memo, val| val.powi(2) + memo).sqrt())
        .collect_into_vec(&mut you_norms);

    // This gives us a me_rows X you_rows matrix
    let mut dot = me.dot(&you.t());
    let norm_dot = ArrayView::from_shape([me_shape[0], 1], &me_norms)?
        .dot(&ArrayView::from_shape([1, you_shape[0]], &you_norms)?);

    Ok(dot / norm_dot)
}

fn analyze_mfccs(sample_rate: f64, samples: &[f64]) -> Vec<f64> {
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
    v
}
