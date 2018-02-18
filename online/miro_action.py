#!/usr/bin/env python
#	@file
#	@section COPYRIGHT
#	Copyright (C) 2017 Consequential Robotics (CqR)
#
#	@section AUTHOR
#	Consequential Robotics http://consequentialrobotics.com
#
#	@section LICENSE
#	For a full copy of the license agreement, see LICENSE.MDK in
#	the MDK root directory.
#
#	Subject to the terms of this Agreement, Consequential Robotics
#	grants to you a limited, non-exclusive, non-transferable license,
#	without right to sub-license, to use MIRO Developer Kit in
#	accordance with this Agreement and any other written agreement
#	with Consequential Robotics. Consequential Robotics does not
#	transfer the title of MIRO Developer Kit to you; the license
#	granted to you is not a sale. This agreement is a binding legal
#	agreement between Consequential Robotics and the purchasers or
#	users of MIRO Developer Kit.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#	OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#	NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#	HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#	WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#	OTHER DEALINGS IN THE SOFTWARE.
#
#	@section DESCRIPTION
#

################################################################

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
from geometry_msgs.msg import Twist

import miro_msgs
from miro_msgs.msg import platform_config,platform_sensors,platform_state,platform_mics,platform_control,core_state,core_control,core_config,bridge_config,bridge_stream

import math
import numpy
import time
import sys
from miro_constants import miro
################################################################

def fmt(x, f):
    s = ""
    x = bytearray(x)
    for i in range(0, len(x)):
        if not i == 0:
            s = s + ", "
        s = s + f.format(x[i])
    return s

def hex2(x):
    return "{0:#04x}".format(x)

def hex4(x):
    return "{0:#06x}".format(x)

def hex8(x):
    return "{0:#010x}".format(x)

def flt3(x):
    return "{0:.3f}".format(x)

def error(msg):
    print(msg)
    sys.exit(0)

def usage():
    print """
Usage:
    miro_ros_client.py robot=<robot_name>

    Without arguments, this help page is displayed. To run the
    client you must specify at least the option "robot".

Options:
    robot=<robot_name>
        specify the name of the miro robot to connect to,
        which forms the ros base topic "/miro/<robot_name>".
        there is no default, this argument must be specified.
    """
    sys.exit(0)

################################################################

class miro_ros_client:

    # Definition of our callback. This loop also calls rospy.loginfo(str), which performs triple-duty:
    # the messages get printed to screen, it gets written to the Node's log file, and it gets written to rosout
    def callback_lstm(self, object):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", object)
        self.drive_pattern = object.data

    def callback_platform_sensors(self, object):


        # ignore until active
        if not self.active:
            return

        # store object
        self.platform_sensors = object

        # send downstream command, ignoring upstream data
        q = platform_control()

        # timing
        sync_rate = 50# syncRate is at 50Hz as set by the creators (this can be changed, from bridge)
        period = 2 * sync_rate # two seconds per period
        z = self.count / period # int/int is resulting in an int

        # advance pattern
        if not z == self.z_bak:
            self.z_bak = z # We enter into this 'if' after every 2 seconds

		

            # create body_vel for next pattern segment
            self.body_vel = Twist()
            if self.drive_pattern == "CarTopBottom":
                print "drive forward"
                self.body_vel.linear.x = +200

            elif self.drive_pattern == "CarBottomTop":
                print "drive backward"
                self.body_vel.linear.x = -200

            elif self.drive_pattern == "PatBody":
                print "drive clockwise"
                self.body_vel.linear.x = +200
                self.body_vel.angular.z = +0.7854


            elif self.drive_pattern == "FixedBody":
                print "Stop"
                self.body_vel.angular.x = 0
                self.body_vel.angular.y = 0
                self.body_vel.angular.z = 0
                self.body_vel.linear.x  = 0
                self.body_vel.linear.y  = 0
                self.body_vel.linear.z  = 0

            elif self.drive_pattern == "PatHead":
                print "turn head"
                if (z == 0 or z==2):
                    q.body_config[2] = -1.0
                    q.body_config_speed[2] = miro.MIRO_P2U_W_LEAN_SPEED_INF
                if (z == 1):
                    q.body_config[2] = +1.0
                    q.body_config_speed[2] = miro.MIRO_P2U_W_LEAN_SPEED_INF

            elif self.drive_pattern == "FixedHead":
                print "tilt head"
                q.body_config[1] = 0
                q.body_config[2] = 0
                q.body_config_speed[1] = miro.MIRO_P2U_W_LEAN_SPEED_INF
                q.body_config_speed[2] = miro.MIRO_P2U_W_LEAN_SPEED_INF
            else:
		
                self.body_vel.angular.x = 0
                self.body_vel.angular.y = 0
                self.body_vel.angular.z = 0
                self.body_vel.linear.x  = 0
                self.body_vel.linear.y  = 0
                self.body_vel.linear.z  = 0
            # else:
            #     # do-si-do
            #     if z == 0:
            #         print "turn left"
            #         self.body_vel.angular.z = +1.5708
            #     if z == 1:
            #         print "drive forward"
            #         self.body_vel.linear.x = +100
            #     if z == 2:
            #         print "turn right"
            #         self.body_vel.angular.z = -1.5708
            #     if z == 3:
            #         print "drive forward"
            #         self.body_vel.linear.x = +100

		# point cameras down


        # publish
        q.body_vel = self.body_vel
        self.pub_platform_control.publish(q)

        # count
        self.count = self.count + 1
        if self.count == 200:
            self.count = 0


	
    def callback_platform_state(self, object):
        
        # ignore until active
        if not self.active:
            return

        # store object
        self.platform_state = object

    def callback_platform_mics(self, object):
        
        # ignore until active
        if not self.active:
            return

        # store object
        self.platform_mics = object

    def callback_core_state(self, object):
        
        # ignore until active
        if not self.active:
            return

        # store object
        self.core_state = object


	


    def loop(self):
        while True:
            if rospy.core.is_shutdown():
                break
            time.sleep(1)
            print "tick"
            if rospy.core.is_shutdown():
                break
            time.sleep(1)
            print "tock"

    def __init__(self):

        # report
        print("initialising...")
        print(sys.version)

        # default data
        self.platform_sensors = None

        # no arguments gives usage
        if len(sys.argv) == 1:
            usage()

        # options
        self.robot_name = ""
        self.drive_pattern = ""

        # handle args
        for arg in sys.argv[1:]:
            f = arg.find('=')
            if f == -1:
                key = arg
                val = ""
            else:
                key = arg[:f]
                val = arg[f+1:]
            if key == "robot":
                self.robot_name = val
            # elif key == "drive":
            #    self.drive_pattern = val
            else:
                error("argument not recognised \"" + arg + "\"")

        # check we got at least one
        if len(self.robot_name) == 0:
            error("argument \"robot\" must be specified")

        # pattern
        self.count = 0
        self.z_bak = -1
        self.body_vel = None

        # set inactive
        self.active = False

        # topic root
        topic_root = "/miro/" + self.robot_name
        print "topic_root", topic_root

        # publish
        self.pub_platform_control = rospy.Publisher(topic_root + "/platform/control",
                    platform_control, queue_size=0)

        # subscribe
        self.sub_sensors = rospy.Subscriber(topic_root + "/platform/sensors",
                platform_sensors, self.callback_platform_sensors)

        self.sub_lstm = rospy.Subscriber("classifyGesture", String, self.callback_lstm)             # This subscriber will receive the data from the topic classifyGesture in this callback_lstm

        # set active
        self.active = True

if __name__ == "__main__":
    rospy.init_node("miro_ros_client_py")
    main = miro_ros_client()
    main.loop()
