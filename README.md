# VideoMimic

[[project page]](https://www.videomimic.net/) [[arxiv]](https://arxiv.org/pdf/2505.03729)  

**Visual Imitation Enables Contextual Humanoid Control. arXiV, 2025.**
    
<div style="background-color: #333; padding: 16px 20px; border-radius: 8px; color: #eee; font-family: sans-serif; line-height: 1.6;">
  <div style="font-size: 14px; margin-bottom: 12px;">
    Arthur Allshire<sup>*</sup>, Hongsuk Choi<sup>*</sup>, Junyi Zhang<sup>*</sup>, David McAllister<sup>*</sup>, 
    Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, Angjoo Kanazawa (*Equal contribution) 
  </div>    
  <div style="font-size: 14px;">
    <i>University of California, Berkeley</i>
  </div>
</div>

## Updates

- **Sep 15, 2025:** Simulation code and preliminary sim2real code released.
- **Jul 6, 2025:** Initial real-to-sim pipeline release. 

## TODO

- [x] Release real‑to‑sim pipeline (July 15th, 2025)
- [x] Release the video dataset (July 15th, 2025) 
- [x] Release simulation pipeline (September 15th, 2025) 
- [x] Release sim2real code (September 15th, 2025) 

# VideoMimic Real-to-Sim

VideoMimic’s [real-to-sim pipeline](real2sim/README.md) reconstructs 3D environments and human motion from single-camera videos and retargets the motion to humanoid robots for imitation learning. It extracts human poses in world coordinates, maps them to robot configurations, and reconstructs environments as pointclouds later converted to meshes.

 
